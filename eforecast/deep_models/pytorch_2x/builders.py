import copy

import numpy as np
import torch
import torch.nn as nn

from eforecast.deep_models.pytorch_2x.layers import layers_func
from eforecast.deep_models.pytorch_2x.global_builders import RBFLayer
from eforecast.deep_models.pytorch_2x.global_builders import apply_activations
from eforecast.deep_models.pytorch_2x.global_builders import act_nan_layer

layers_functions = layers_func()


def get_size(size, layer_id, layers, output_shape):
    if isinstance(size, bool):
        return size
    if isinstance(size, set):
        size = list(size)
    if isinstance(size, list):
        size = size[0]
    if size == 'linear':
        if layer_id + 1 >= len(layers):
            raise ValueError('Cannot be linear the last layer')
        size_next = layers[layer_id + 1][1]
        if isinstance(size_next, set):
            size_next = list(size_next)
        if isinstance(size_next, list):
            size_next = size_next[0]
        return int((size_next + np.prod(output_shape)) / 2)
    elif size < 8:
        return int(size * np.prod(output_shape))
    else:
        return size


def get_output_shape(model, image_dim):
    shapes = list(model(torch.rand(10, *image_dim[1:])).data.shape)
    return shapes


class build_layer(nn.Module):
    def __init__(self, shape, layers, name_scope, params, train=True):
        super(build_layer, self).__init__()
        output_shape = shape
        print(f'Graph of {name_scope} building')
        self.layers_built = nn.ModuleList()
        for layer_id, layer_tuple in enumerate(layers):
            layer_name, size = layer_tuple
            print(f'Input has shape {output_shape}')
            if len(output_shape) == 3 and '3d' in layer_name:
                layer_tuple = ('conv_2d', size)
                layer_name, size = layer_tuple
            if layer_name == 'dense':
                if isinstance(output_shape, dict):
                    output_shape = list(output_shape.values())[0]
                size = get_size(size, layer_id, layers, output_shape[1:])
            if isinstance(size, set):
                size = list(size)
            if isinstance(size, list):
                if len(size) > 0:
                    size = size[0]

            if layer_name not in {'Flatten', 'Dropout', 'Reshape', 'concatenate'}:
                if layer_name == 'lstm':
                    params['is_lstm_output'] = True if layers[layer_id + 1][0] == 'Flatten' else False
                layer_ = layers_functions[layer_name](output_shape, params, size, name_scope,
                                                      str(layer_id), train=train)
                self.layers_built.append(layer_)
            elif layer_name == 'Reshape':
                lstm_layers = [l for l in self.layers_built if isinstance(l, layers_functions['lstm'])][-1]
                lstm_lags = lstm_layers.lstm.input_size
                layer_ = layers_functions[layer_name]([lstm_lags, int(output_shape[1] / lstm_lags)],
                                                      str(layer_id))
                self.layers_built.append(layer_)
            elif layer_name == 'concatenate':
                layer_ = layers_functions[layer_name](output_shape, str(layer_id))
                self.layers_built.append(layer_)
            elif layer_name == 'Dropout':
                layer_ = nn.Dropout(size)
                self.layers_built.append(layer_)
            elif layer_name == 'Flatten':
                layer_ = nn.Flatten()
                self.layers_built.append(layer_)
            else:
                names = [name for name in layers_functions.keys()]
                raise ValueError(f"Unknown layer name {layer_name}. Valid names {names}")
            output_shape = layer_.output_shape if hasattr(layer_, 'output_shape') \
                else get_output_shape(layer_, output_shape)
        self.output_shape = output_shape

    def forward(self, x):
        for layer in self.layers_built:
            x = layer(x)
        return x


class proba_output(nn.Module):
    def __init__(self, shape, quantiles):
        super(proba_output, self).__init__()
        self.shape = shape
        self.quantiles = quantiles
        self.layers = nn.ModuleDict({f"model_output_{i}_q{q}": nn.Linear(shape[1], 1)
                                     for i, q in enumerate(quantiles)})

    def string(self):
        print(f"layer {self.name} has input shape {self.shape}")
        print(f"layer weights {self.name} has shape {list(self.dense.weight.size())}")

    def forward(self, x):
        output = torch.tensor([])
        for i, q in enumerate(self.quantiles):
            yq = self.layers[f"model_output_{i}_q{q}"](x)
            yq = yq.unsqueeze(1)
            output = torch.cat((output, yq), 1)
        return output


def get_shapes(x):
    shapes = dict()
    if isinstance(x, dict):
        for name, inp in sorted(x.items()):
            if 'act' not in name and 'clustering' not in name:
                shapes[name] = get_shapes(inp)
    else:
        shapes = list(x.shape)
    return shapes


def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)

class build_graph(nn.Module):
    def __init__(self, x, model_layers, params, is_fuzzy=False, train=True,
                 probabilistic=False, quantiles=None, device='cpu'):
        super(build_graph, self).__init__()
        self.act_nans = 0
        self.params = params
        self.device = device
        self.is_fuzzy = is_fuzzy
        self.output_shapes = dict()
        self.model_layers_built = nn.ModuleDict()
        self.non_input_branch_names = set()
        if is_fuzzy:
            self.thres_act = self.params['thres_act']
            self.thres_act_tf = torch.Tensor([self.thres_act]).squeeze().to(self.device)
            self.thres_act_tf_up = torch.Tensor([self.thres_act + self.thres_act / 10]).squeeze().to(self.device)
            self.thres_act_tf_ml = torch.Tensor([10 / self.thres_act]).squeeze().to(self.device)
            self.model_layers_built['clustering'] = RBFLayer(params)
            self.model_layers_built['act_nan'] = act_nan_layer(params['thres_act'])
            self.non_input_branch_names.add(f'clustering')
            self.non_input_branch_names.add(f'act_nan')
            for rule in sorted(params['rules']):
                self.model_layers_built[f'apply_act_{rule}'] = apply_activations(params['thres_act'])
                self.non_input_branch_names.add(f'apply_act_{rule}')
        shapes = get_shapes(x)
        name_scope_list = []
        for name_scope in sorted(shapes.keys()):
            shape = shapes[name_scope]
            if name_scope in {'output'}:
                continue
            layer = model_layers[name_scope]

            params_temp = copy.deepcopy(params)

            prime_layers = build_layer(shape, layer, name_scope, params_temp, train=train)

            self.output_shapes[name_scope] = prime_layers.output_shape
            self.model_layers_built[name_scope] = prime_layers
            name_scope_list.append(name_scope)

        self.connections = []
        group_layer_names = set(['/'.join(name_split.split('/')[:-1]) for name_split in name_scope_list
                                 if '/'.join(name_split.split('/')[:-1]) != ''])
        branch_layer_names = set([name_split.split('/')[-1] for name_split in name_scope_list])
        while len(group_layer_names) > 0:
            new_name_scope_list = []
            group_connections = dict()
            for group_name in sorted(group_layer_names):
                model_list = []
                shape_list = dict()
                for name_branch in sorted(name_scope_list):
                    if group_name + '/' in name_branch:
                        model_list.append(name_branch)
                        shape_list[name_branch] = self.output_shapes[name_branch]
                group_connections[group_name] = model_list
                layers_name = 'output' if 'hidden_layer' not in model_layers.keys() else 'hidden_layer'
                layers_branch_built = build_layer(shape_list, model_layers[layers_name], group_name,
                                                  params, train=train)
                self.model_layers_built[group_name] = layers_branch_built
                self.output_shapes[group_name] = layers_branch_built.output_shape
                self.non_input_branch_names.add(group_name)
                new_name_scope_list.append(group_name)
            name_scope_list = copy.deepcopy(new_name_scope_list)
            group_layer_names = set(['/'.join(name_split.split('/')[:-1]) for name_split in name_scope_list
                                     if '/'.join(name_split.split('/')[:-1]) != ''])
            branch_layer_names = set([name_split.split('/')[-1] for name_split in name_scope_list])
            self.connections.append(group_connections)

        if len(branch_layer_names) > 0:
            model_list = []
            shape_list = dict()
            for name_branch in sorted(name_scope_list):
                model_list.append(name_branch)
                shape_list[name_branch] = self.output_shapes[name_branch]
            self.connections.append({'hidden_output': model_list})
            name_scope = 'hidden_output'
            self.non_input_branch_names.add(name_scope)
            layer_output = build_layer(shape_list, model_layers['output'],
                                       name_scope, params, train=train)
            self.model_layers_built[name_scope] = layer_output
            self.output_shapes[name_scope] = layer_output.output_shape

        name_scope = 'output'
        self.non_input_branch_names.add(name_scope)
        if not probabilistic:
            shape = self.output_shapes['hidden_output']
            layer_output = layers_functions['dense'](shape, {'act_func': None}, params['n_out'],
                                                     name_scope,
                                                     'output', train=train)
            self.model_layers_built['output'] = layer_output
        else:
            shape = self.output_shapes['hidden_output']
            self.model_layers_built['output'] = proba_output(shape, quantiles)

    def forward(self, x, get_activations=False):
        self.act_nans = 0
        activations = None
        outputs = dict()
        if self.is_fuzzy:
            for key in self.model_layers_built.keys():
                if key in {'clustering'}:
                    if 'clustering' not in self.model_layers_built.keys() or 'clustering' not in x.keys():
                        raise ValueError('Since the model is fuzzy, clustering layers should include to model layers or'
                                         'clustering inputs should inlude inside x inputs')
                    activations = self.model_layers_built['clustering'](x['clustering'].to(self.device))
                    if get_activations:
                        return activations
                    activations = torch.mul(self.thres_act_tf_ml,
                                            torch.sub(torch.clip(activations, self.thres_act_tf, self.thres_act_tf_up),
                                                      self.thres_act_tf))
                    self.act_nans = self.model_layers_built['act_nan'](activations)

        for key in self.model_layers_built.keys():
            if key not in self.non_input_branch_names:
                if isinstance(x[key], dict):
                    for key1 in x[key].keys():
                        x[key][key1] = x[key][key1].to(self.device)
                else:
                    x[key] = x[key].to(self.device)
                if self.is_fuzzy and activations is not None and 'rule' in key.split('/')[-1]:
                    rule = key.split('/')[-1]
                    n_rule = int(key.split('_')[-1])
                    out_branch = self.model_layers_built[key](x[key])
                    outputs[key] = self.model_layers_built[f'apply_act_{rule}'](out_branch,
                                                                                       activations[:, n_rule].unsqueeze(1))
                else:
                    outputs[key] = self.model_layers_built[key](x[key])


        if len(self.connections) > 0:
            for group_connection in self.connections:
                for name_group, model_list in group_connection.items():
                    if 'apply_act_rule' not in name_group and name_group not in {'clustering', 'output'}:
                        input_brunch = [outputs[key] for key in model_list]
                        if len(input_brunch) == 1:
                            input_brunch = input_brunch[0]
                        outputs[name_group] = self.model_layers_built[name_group](input_brunch)

        name_scope = 'output'
        return self.model_layers_built[name_scope](outputs['hidden_output'])
