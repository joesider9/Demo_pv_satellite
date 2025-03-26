import numpy as np
import tensorflow as tf
from einops import rearrange
from einops import repeat
from einops import einsum
from eforecast.deep_models.tf_2x.transformers.models.auto_model import AutoModel
from eforecast.deep_models.tf_2x.unet import models

act_funcs = {'elu': tf.nn.elu,
             'sigmoid': tf.nn.sigmoid,
             'relu': tf.nn.relu,
             'gelu': tf.nn.gelu,
             'tanh': tf.nn.tanh}

class conv_2d(object):
    def __init__(self, shape, params, size, name_scope, layer_id, train=True):
        super(conv_2d, self).__init__()
        if len(shape) == 3:
            shape.insert(1, 1)
        self.shape = shape
        self.size = size
        x_shape1 = shape
        x_shape1 = np.array(x_shape1[-3:-1])
        x_shape = x_shape1 // size
        x_shape[x_shape <= 1] = 2
        x_shape = np.minimum(x_shape, x_shape1)
        kernels = x_shape.tolist()
        self.act_func = act_funcs[params['act_func']]
        x_shape = x_shape // 2
        x_shape[x_shape == 0] = 1
        pool_size = x_shape.tolist()
        pool_size = [int(p) for p in pool_size]
        kernels = [int(k) for k in kernels]
        self.name = f'{name_scope}_conv_2d_{layer_id}'
        self.conv = tf.keras.layers.Conv2D(shape[1], int(params['filters']),
                              kernel_size=tuple(kernels[-2:]),
                              padding="valid")

        self.pool = tf.keras.layers.AveragePooling2D(tuple(pool_size[-2:]), stride=1)

    def __call__(self, x):
        x_shape1 = list(x.size())
        if len(x_shape1) == 3:
            x = tf.expand_dims(x, axis=1)
        return self.pool(self.act_func(self.conv(x)))


class conv_3d(object):
    def __init__(self, shape, params, size, name_scope, layer_id, train=True):
        super(conv_3d, self).__init__()
        if len(shape) == 3:
            shape.insert(1, 1)
        if len(shape) == 4:
            shape.insert(1, 1)
        self.shape = shape
        self.size = size
        x_shape1 = shape
        x_shape1 = np.array(x_shape1[-3:-1])
        x_shape = x_shape1 // size
        x_shape[x_shape <= 2] = 3
        x_shape = np.minimum(x_shape, x_shape1)
        kernels = x_shape.tolist()
        if len(kernels) != 3:
            kernels = kernels + [1]
        self.act_func = act_funcs[params['act_func']]
        x_shape = x_shape // 2
        x_shape[x_shape == 0] = 1
        pool_size = x_shape.tolist()
        if len(pool_size) != 3:
            pool_size = pool_size + [1]
        pool_size = [int(p) for p in pool_size]
        kernels = [int(k) for k in kernels]
        self.name = f'{name_scope}_conv_3d_{layer_id}'
        self.conv = tf.keras.layers.Conv3D(shape[1], int((int(layer_id) + 1) * params['filters']),
                              kernel_size=tuple(kernels[-3:]),
                              stride=2,
                              padding="valid")

        self.pool = tf.keras.layers.AveragePooling3D(tuple(pool_size[-3:]), stride=1)
    def __call__(self, x):
        if len(x.size()) == 3:
            x = tf.expand_dims(x, axis=1)
        if len(x.size()) == 4:
            x = tf.expand_dims(x, axis=1)
        return self.pool(self.act_func(self.conv(x)))


class time_distr_conv_2d(object):
    def __init__(self, shape, params, size, name_scope, layer_id, train=True):
        super(time_distr_conv_2d, self).__init__()
        if len(shape) == 3:
            shape.insert(1, 1)
        if len(shape) == 4:
            shape.insert(1, 1)
        self.shape = shape
        self.size = size
        self.act_func = act_funcs[params['act_func']]
        self.name = f'{name_scope}_time_distr_{layer_id}'
        shape_conv = [s for ii, s in enumerate(shape) if ii != 1]
        self.conv_list = []
        for i in range(shape[1]):
            self.conv_list.append(conv_2d(shape_conv, params, size, self.name_conv, i))

    def __call__(self, x):
        if list(x.size()) == 3:
            x = tf.expand_dims(x, axis=1)
        if list(x.size()) == 4:
            x = tf.expand_dims(x, axis=1)
        batch_size, time_steps, C, H, W = x.size()
        output = tf.tensor([])
        for i in range(time_steps):
            output_t = self.conv_list[i](x[:, i, ...])
            output_t = tf.expand_dims(output_t, axis=1)
            output = tf.keras.layers.concatenate((output, output_t), 1)
        return output


class time_distr_conv_3d(object):
    def __init__(self, shape, params, size, name_scope, layer_id, train=True):
        super(time_distr_conv_3d, self).__init__()
        if len(shape) == 3:
            shape.insert(1, 1)
        if len(shape) == 4:
            shape.insert(1, 1)
        self.shape = shape
        self.size = size
        self.act_func = act_funcs[params['act_func']]
        self.name = f'{name_scope}_time_distr_{layer_id}'
        shape_conv = [s for ii, s in enumerate(shape) if ii != 1]
        self.conv_list = []
        for i in range(shape[1]):
            self.conv_list.append(conv_3d(shape_conv, params, size, self.name_conv, i))

    def __call__(self, x):
        if list(x.size()) == 3:
            x = tf.expand_dims(x, axis=1)
        if list(x.size()) == 4:
            x = tf.expand_dims(x, axis=1)
        time_steps = x.size()[1]
        output = tf.tensor([])
        for i in range(time_steps):
            output_t = self.conv_list[i](x[:, i, ...])
            output_t = output_t.unsqueeze(1)
            output = tf.keras.layers.concatenate((output, output_t), 1)
        return output


class lstm(object):
    def __init__(self, shape, params, size, name_scope, layer_id, train=True):
        super(lstm, self).__init__()
        self.shape = shape
        self.size = size
        self.act_func = act_funcs[params['act_func']]
        self.name = f'{name_scope}_lstm_{layer_id}'
        if isinstance(shape, dict):
            B, Tpred, Ctime = shape['future_calendar']
            self.obs_embed = RnnEmbedding(shape['past_obs'], Tpred, layer_id)
            future_calendar = tf.keras.layers.concatenate([tf.random.normal(shape['past_calendar']),
                                         tf.random.normal(shape['future_calendar'])], 1)
            future_data = tf.keras.layers.concatenate([tf.random.normal(shape['past_data']),
                                    tf.random.normal(shape['future_data'])], 1)

            past_obs = tf.random.normal(shape['past_obs'])
            future_obs = tf.keras.layers.concatenate([past_obs, self.obs_embed(past_obs)], 1)
            x_dec = tf.keras.layers.concatenate([future_obs, future_data, future_calendar], -1)
            self.enc_in = x_dec.shape[-1]
        else:
            B = shape[0]
            x_dec = tf.random.normal(shape)
            self.enc_in = shape[-1]
        self.lstm = tf.keras.layers.LSTM(
        int(size * self.enc_in),
        activation='tanh',
        recurrent_activation='sigmoid',
        name=self.name,
        return_sequences=True,
        unroll=False,
        use_bias=True,
        recurrent_dropout=0)
        lstm_out = self.lstm(x_dec)
        self.output_shape = self.act_func(lstm_out).shape

    def string(self):
        print(f"layer {self.name} has input shape {self.shape}")
        print(f"layer weights {self.name} has shape {list(self.lstm.weight.size())}")

    def __call__(self, x):
        if isinstance(x, dict):
            future_calendar = tf.keras.layers.concatenate([x['past_calendar'], x['future_calendar']], 1)
            future_data = tf.keras.layers.concatenate([x['past_data'], x['future_data']], 1)
            past_obs = x['past_obs']
            device = past_obs.device
            B, Tpred, Ctime = future_calendar.shape

            future_obs = tf.keras.layers.concatenate([past_obs, self.obs_embed(past_obs)], 1)
            inp = tf.keras.layers.concatenate([future_obs, future_data, future_calendar], -1)
        else:
            B = x.shape[0]
            inp = x
            device = x.device
        hidden_state = (tf.random.normal(1, B, int(self.size * self.enc_in)).to(device),
                        tf.random.normal(1, B, int(self.size * self.enc_in)).to(device))
        lstm_out = self.lstm(inp, hidden_state)[0]
        return self.act_func(lstm_out)


class transformer(object):
    def __init__(self, shape, params, size, name_scope, layer_id, train=True):
        super(transformer, self).__init__()
        B, Tpred, Ctime = shape['future_calendar']
        _, _, Cdata = shape['future_data']
        _, Tpast, _ = shape['past_calendar']
        _, _, Cobs = shape['past_obs']
        self.shape = shape
        self.act_func = act_funcs[params['act_func']]
        self.name = f'{name_scope}_transformer_{layer_id}'
        self.transformer_name = params['transformer_name']
        enc_in = Cdata + Cobs
        dec_in = Cdata + Cobs
        self.size = size if self.transformer_name not in {'Autoformer'} else enc_in
        self.transfromer_model = AutoModel(params["transformer_type"], shape, params, name=self.name)

        future_calendar = tf.keras.layers.concatenate([tf.random.normal(shape['past_calendar']),
                                     tf.random.normal(shape['future_calendar'])], 1)
        future_data = tf.keras.layers.concatenate([tf.random.normal(shape['past_data']),
                                tf.random.normal(shape['future_data'])], 1)

        past_calendar = tf.random.normal(shape['past_calendar'])
        past_data = tf.random.normal(shape['past_data'])
        past_obs = tf.random.normal(shape['past_obs'])
        x_enc = tf.keras.layers.concatenate([past_obs, past_data], -1)
        x_mark_enc = past_calendar
        x_dec = future_data
        x_mark_dec = future_calendar
        self.output_shape = self.transfromer_model(x_enc, x_mark_enc, x_dec, x_mark_dec).shape


    def __call__(self, x):
        future_calendar = tf.keras.layers.concatenate([x['past_calendar'], x['future_calendar']], 1)
        future_data = tf.keras.layers.concatenate([x['past_data'], x['future_data']], 1)
        past_calendar = x['past_calendar']
        past_data = x['past_data']
        past_obs = x['past_obs']
        future_obs = tf.keras.layers.concatenate([past_obs, self.obs_embed(past_obs)], 1)
        x_enc = tf.keras.layers.concatenate([past_obs, past_data], -1)
        x_mark_enc = past_calendar
        x_dec = tf.keras.layers.concatenate([future_obs, future_data], -1)
        x_mark_dec = future_calendar

        return self.act_func(self.transfromer_model(x_enc, x_mark_enc, x_dec, x_mark_dec))

class dense(object):
    def __init__(self, shape, params, size, name_scope, layer_id, train=True):
        super(dense, self).__init__()
        self.shape = shape
        self.size = size
        if params['act_func'] is not None:
            self.act_func = act_funcs[params['act_func']]
        else:
            self.act_func = None
        self.name = f'{name_scope}_dense_{layer_id}'
        self.dense = tf.keras.layers.Dense(units=int(size), activation=self.act_func)
        self.output_shape = self.dense(tf.random.normal(self.shape)).shape

    def __call__(self, x):
        if self.act_func is not None:
            return self.act_func(self.dense(x))
        else:
            return self.dense(x)


class Reshape(object):
    def __init__(self, shape, layer_id):
        super(Reshape, self).__init__()
        self.shape = shape
        self.name = f'Reshape_{layer_id}'

    def __call__(self, x):
        return rearrange(x, 'b (c h) -> b c h', c=self.shape[0], h=self.shape[1])


class Concat_(object):
    def __init__(self, shape, layer_id):
        super(Concat_, self).__init__()
        self.shape = shape
        self.name = f'Concat_{layer_id}'

    def __call__(self, x):
        return tf.keras.layers.concatenate(x, -1)

def layers_func():
    layers = {'conv_2d': conv_2d,
              'time_distr_conv_2d': time_distr_conv_2d,
              'conv_3d': conv_3d,
              'time_distr_conv_3d': time_distr_conv_3d,
              'lstm': lstm,
              'transformer': transformer,
              'Reshape': Reshape,
              'unet': unet,
              # 'yolo': yolo,
              'timm_net': timm_net,
              'cross_attention': CrossAttention,
              'concatenate': Concat_,
              'vit_net': vit_net,
              'time_distr_vit_net': time_distr_vit_net,
              'dense': dense
              }
    return layers


def layers_func():
    layers = {'conv_2d': conv_2d,
              'time_distr_conv_2d': time_distr_conv_2d,
              'conv_3d': conv_3d,
              'time_distr_conv_3d': time_distr_conv_3d,
              'lstm': lstm,
              'transformer': transformer,
              'unet': unet,
              'dense': dense
              }
    return layers
