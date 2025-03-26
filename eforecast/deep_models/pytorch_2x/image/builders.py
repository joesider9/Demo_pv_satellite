import copy

import numpy as np
import torch
import torch.nn as nn

from eforecast.deep_models.pytorch_2x.image.layers import layers_func

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
                    lstm_lags = output_shape[1]
                layer_ = layers_functions[layer_name](output_shape, params, size, name_scope,
                                                      str(layer_id), train=train)
                self.layers_built.append(layer_)
            elif layer_name == 'Reshape':
                layer_ = layers_functions[layer_name]([lstm_lags, int(output_shape[2] / lstm_lags)],
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
                shape = list(inp.shape)
                shapes[name] = shape
    else:
        shape = list(x.shape)
        shapes['input'] = shape
    return shapes


def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)

class build_graph(nn.Module):
    def __init__(self, x, model_layers, params, train=True,
                 probabilistic=False, quantiles=None, device='cpu'):
        super(build_graph, self).__init__()
        self.params = params
        self.device = device
        shapes = get_shapes(x)
        name_scope_list = []
        self.output_shapes = dict()
        self.model_layers_built = nn.ModuleDict()
        for name_scope in sorted(model_layers.keys()):
            if name_scope in {'output'}:
                continue
            layer = model_layers[name_scope]
            params_temp = copy.deepcopy(params)

            prime_layers = build_layer(shapes,
                                       layer,
                                       name_scope,
                                       params_temp,
                                       train=train)
            self.output_shapes[name_scope] = prime_layers.output_shape
            self.model_layers_built[name_scope] = prime_layers
            name_scope_list.append(name_scope)

        group_layers_build = self.model_layers_built
        self.connections = []
        if len(params['group_layers']) > 1 and 'input' not in params['group_layers']:
            while True:
                name_scope_list = list(set(name_scope_list))
                new_branch_dict = dict()
                new_name_scope_list = []
                group_names = set()
                for name_branch in sorted(name_scope_list):
                    name_split = name_branch.split('_')
                    if name_split[0] == 'rule':
                        name_split = ['_'.join(name_split[:2])] + name_split[2:]
                    if 'input' in name_split:
                        raise ValueError(
                            "keyword 'input' is not allowed to name a group or name_scope when have more than "
                            "one group_layer")
                    if len(name_split) > 1:
                        new_name_scope = '_'.join(name_split[:-1])
                        new_name_scope_list.append(new_name_scope)
                        group_names.add(new_name_scope)
                if len(new_name_scope_list) > 0:
                    name_scope_list = copy.deepcopy(new_name_scope_list)
                else:
                    break
                group_connections = dict()
                for group_name in sorted(group_names):
                    model_list = []
                    for name_branch in sorted(group_layers_build.keys()):
                        if group_name + '_' in name_branch:
                            model_list.append(name_branch)
                    group_connections[group_name] = model_list
                    shape1 = 0
                    for name_ in model_list:
                        shape1 += self.output_shapes[name_][1]
                        shape = self.output_shapes[name_]
                    shape[1] = shape1
                    layers_branch_built = build_layer(shape, model_layers['output'], group_name,
                                                      params, train=train)
                    self.model_layers_built[group_name] = layers_branch_built
                    self.output_shapes[group_name] = layers_branch_built.output_shape
                    new_branch_dict[group_name] = layers_branch_built
                if len(new_branch_dict) > 0:
                    group_layers_build = new_branch_dict
                self.connections.append(group_connections)
            group_names = list(group_names)
        else:
            group_names = name_scope_list
        model_list = []
        for group_name in group_names:
            model_list.append(group_name)


        if len(model_list) > 0:
            self.connections.append({'hidden_output': model_list})
        else:
            raise RuntimeError('Failed to build model output')

        name_scope = 'hidden_output'
        layer_output = build_layer(self.output_shapes, model_layers['output'],
                                   name_scope, params, train=train)
        self.model_layers_built[name_scope] = layer_output
        self.output_shapes[name_scope] = layer_output.output_shape
        name_scope = 'output'
        shape = self.output_shapes['hidden_output']
        layer_output = layers_functions['dense'](shape, {'act_func': None}, params['n_out'],
                                                 name_scope,
                                                 'output', train=train)
        self.model_layers_built['output'] = layer_output
        if 'row_data' in self.output_shapes.keys():
            name_scope = 'out_row_data'
            shape = self.output_shapes['row_data']
            layer_output = layers_functions['dense'](shape, {'act_func': None}, params['n_out'],
                                                     name_scope,
                                                     'out_row_data', train=train)
            self.model_layers_built['out_row_data'] = layer_output
        if 'images' in self.output_shapes.keys():
            name_scope = 'out_images'
            shape = self.output_shapes['images']
            layer_output = layers_functions['dense'](shape, {'act_func': None}, params['n_out'],
                                                     name_scope,
                                                     'out_images', train=train)
            self.model_layers_built['out_images'] = layer_output


    def forward(self, x, get_activations=False):
        outputs = dict()
        for key in self.model_layers_built.keys():
            if key not in {'hidden_output', 'output', 'out_images', 'out_row_data'}:
                outputs[key] = self.model_layers_built[key](x)

        if len(self.connections) > 0:
            for group_connection in self.connections:
                for name_group, model_list in group_connection.items():
                    if name_group not in {'hidden_output', 'output'}:
                        input_brunch = [outputs[key] for key in sorted(model_list)]
                        outputs[name_group] = self.model_layers_built[name_group](input_brunch)

        if len(self.connections) > 0:
            for group_connection in self.connections:
                for name_group, model_list in group_connection.items():
                    if name_group == 'hidden_output':
                        input_brunch = [outputs[key] for key in sorted(model_list)]
                        out_brunch = {key: outputs[key] for key in sorted(model_list)}
                        if len(input_brunch) == 1:
                            input_brunch = input_brunch[0]
                        outputs[name_group] = self.model_layers_built[name_group](input_brunch)

        name_scope = 'output'
        output = self.model_layers_built[name_scope](outputs['hidden_output'])
        if 'out_images' in self.model_layers_built.keys():
            out_images = self.model_layers_built['out_images'](out_brunch['images'])
        else:
            out_images = None
        if 'out_row_data' in self.model_layers_built.keys():
            out_row_data = self.model_layers_built['out_row_data'](out_brunch['row_data'])
        else:
            out_row_data = None
        return output, out_images, out_row_data

