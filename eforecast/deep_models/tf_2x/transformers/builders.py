import copy

import numpy as np
import tensorflow as tf

from eforecast.deep_models.tf_2x.layers import layers_func
from eforecast.deep_models.tf_2x.global_builders import RBFLayer
from eforecast.deep_models.tf_2x.global_builders import apply_activations
from eforecast.deep_models.tf_2x.global_builders import act_nan_layer

layers_functions = layers_func()


def get_size(size, layer_id, layers, output_shape):
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


def build_layer(input_name, shape, layers, name_scope, params, train=True):
    if isinstance(shape, dict):
        input_layer = dict()
        input_layer['observations'] = tf.keras.Input(shape=shape['observations'], dtype=tf.float32, name='observations')
        input_layer['future'] = tf.keras.Input(shape=shape['future'], dtype=tf.float32, name='future')
        input_layer['calendar'] = tf.keras.Input(shape=shape['calendar'], dtype=tf.float32, name='calendar')
    else:
        input_layer = tf.keras.Input(shape=shape, dtype=tf.float32, name=input_name)
    output_layer = input_layer
    output_shape = shape
    with tf.name_scope(name_scope) as scope:
        print(f'Graph of {name_scope} building')
        layers_built = dict()
        for layer_id, layer_tuple in enumerate(layers):
            layer_name, size = layer_tuple
            print(f'Input has shape {output_shape}')
            if layer_name == 'dense' :
                size = get_size(size, layer_id, layers, output_shape)
            if isinstance(size, set):
                size = list(size)
            if isinstance(size, list):
                if len(size) > 0:
                    size = size[0]

            if layer_name not in {'Flatten', 'Dropout', 'Reshape'}:
                if layer_name == 'lstm':
                    lstm_lags = output_layer.get_shape()[1]
                layer_ = layers_functions[layer_name](input_name, output_shape, params, size, name_scope,
                                                      str(layer_id), train=train)
                layers_built[layer_.name] = layer_
            elif layer_name == 'Reshape':
                layer_ = tf.keras.layers.Reshape([lstm_lags,
                                                  int(output_layer.get_shape()[1] / lstm_lags)],
                                                 name=f'Reshape_{layer_id}')
                layers_built[layer_.name] = layer_
            elif layer_name == 'Dropout':
                layer_ = tf.keras.layers.Dropout(size)
                layers_built[layer_.name] = layer_
            elif layer_name == 'Flatten':
                layer_ = tf.keras.layers.Flatten()
                layers_built[layer_.name] = layer_
            else:
                names = [name for name in layers_functions.keys()]
                raise ValueError(f"Unknown layer name {layer_name}. Valid names {names}")
            output_layer = layer_(output_layer)
            output_shape = output_layer.get_shape().as_list()[1:]
            input_name = layer_.output.name

    model_layer = tf.keras.Model(input_layer, output_layer, name='model_prime_' + name_scope)
    return model_layer, layers_built


def proba_output(shape, quantiles):
    x = tf.keras.Input(tuple(shape[1:]), name='model_output')
    output_layers = []
    outputs = []
    for i, q in enumerate(quantiles):
        # Get output layers
        layer_ = tf.keras.layers.Dense(units=1,
                                       name="model_output_{}_q{}".format(i, int(q * 100)))
        outputs.append(layer_(x))
        output_layers.append(layer_)
    proba_out = tf.keras.Model(x, outputs)
    return proba_out, output_layers


def build_graph(x, model_layers, params, train=True, probabilistic=False, quantiles=None):
    inputs = dict()
    shapes = dict()
    if isinstance(x, dict):
        for name, inp in sorted(x.items()):
            if name in {'observations', 'future', 'calendar'}:
                shape = tuple(inp.shape[1:])
                inputs[name] = tf.keras.Input(shape, dtype=tf.float32, name='input_' + name)
                shapes[name] = shape
            else:
                raise ValueError(f'Found {name} in input dict, the input should'
                                 f' contain only observations, future, calendar')
    else:
        raise ValueError(f'The input of a transformer should be dict with keys: observations, future, calendar')

    input_name = 'transformer'
    name_scope = 'transformer'
    layer = model_layers['input']
    params_temp = copy.deepcopy(params)

    with tf.name_scope(f'prime_{name_scope}') as scope:
        model_layer, prime_layers = build_layer(input_name, shapes,
                                                layer,
                                                f'prime_{name_scope}',
                                                params_temp,
                                                train=train)

    model_output = model_layer(inputs)


    shape = tuple(model_output.get_shape().as_list()[1:])
    name_scope = 'output_scope'
    with tf.name_scope("output_scope") as scope:
        layer_output, layers_built = build_layer('input_output_scope', shape, model_layers['output'],
                                                 name_scope, params, train=train)
        model_output = layer_output(model_output)
    name_scope = 'output'
    if not probabilistic:
        with tf.name_scope(name_scope) as scope:
            shape = tuple(model_output.get_shape().as_list()[1:])
            layer_output = layers_functions['dense']('inp_output', shape, {'act_func': None}, params['n_out'], name_scope,
                                                     'output', train=train)
            model_output = layer_output(model_output)
    else:
        with tf.name_scope(name_scope) as scope:
            shape = tuple(model_output.get_shape().as_list()[1:])
            proba_layer_output, layer_output = proba_output(shape, quantiles)
            model_output = proba_layer_output(model_output)

    global_model = tf.keras.Model(inputs, model_output)
    return global_model

