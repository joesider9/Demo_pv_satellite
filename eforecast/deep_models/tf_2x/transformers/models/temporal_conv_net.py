import tensorflow as tf

from eforecast.deep_models.tf_2x.transformers.layers.embed_layer import RnnEmbedding
from eforecast.deep_models.tf_2x.transformers.layers.embed_layer import DataEmbedding
from eforecast.deep_models.tf_2x.transformers.layers.embed_layer import TokenEmbedding
from eforecast.deep_models.tf_2x.transformers.layers.embed_layer import TimeEmbedding
from eforecast.deep_models.tf_2x.transformers.layers.embed_layer import PositionalEmbedding
from eforecast.deep_models.tf_2x.transformers.layers.cnn_layer import ConvTemp
from eforecast.deep_models.tf_2x.transformers.layers.dense_layer import DenseTemp
from eforecast.deep_models.tf_2x.transformers.layers.tcn_layer import TCN_layer


def which_norm(norm_layer_type):
    use_batch_norm = False
    use_layer_norm = False
    use_weight_norm = False
    if norm_layer_type is None:
        return use_batch_norm, use_layer_norm, use_weight_norm
    if norm_layer_type == 'batch':
        use_batch_norm = True
    elif norm_layer_type == 'layer':
        use_layer_norm = True
    elif norm_layer_type == 'weight':
        use_weight_norm = True
    return use_batch_norm, use_layer_norm, use_weight_norm


class TemporalConvNet(object):

    def __init__(self, shape, params, name='Transformer'):
        self.params = params
        self.use_decoder = params["use_decoder"]
        self.embedding_full = params["embedding_full"]
        self.data_all = params['data_all']
        self.stacks = params["stacks"]
        self.filters = params["filters"]
        self.kernel_size = [params["kernel_size"] for i in range(params["dilations"])]
        self.dilations = [2 ** i for i in range(params["dilations"])]
        self.padding = params["padding"]
        self.use_batch_norm, self.use_layer_norm, self.use_weight_norm = which_norm(params["norm_layer"])
        self.encoder_type = params["encoder_type"]
        self.dropout = params["dropout"]
        self.act_func = params["act_func"]
        self.dense_size = params["dense_size"]
        self.embedding_positional_flag = params['embedding_positional']
        self.embedings_past = RnnEmbedding(shape['future'][0], return_state=True,
                                           name=f'{name}_RnnEmbedding')
        self.embedings_future = TokenEmbedding(shape['future'][0], name=f'{name}_TokenEmbedding')
        self.embedings_calendar = TimeEmbedding(shape['calendar'], name=f'{name}_TimeEmbedding')
        self.embedings_positional = PositionalEmbedding()
        if self.embedding_full == 'discrete':
            self.encode_self = Encoder(
                self.kernel_size, self.dilations, self.filters, self.act_func, causal=False,
                padding=self.padding,
                use_batch_norm=self.use_batch_norm,
                use_layer_norm=self.use_layer_norm,
                use_weight_norm=self.use_weight_norm,
                name=f'{name}_encode_self'
            )
            self.encode_future = Encoder(
                self.kernel_size, self.dilations, self.filters, self.act_func, causal=False,
                padding=self.padding,
                use_batch_norm=self.use_batch_norm,
                use_layer_norm=self.use_layer_norm,
                use_weight_norm=self.use_weight_norm,
                name=f'{name}_encode_future'
            )
        else:
            self.encoder = Encoder(
                self.kernel_size, self.dilations, self.filters, self.act_func, causal=False,
                padding=self.padding,
                use_batch_norm=self.use_batch_norm,
                use_layer_norm=self.use_layer_norm,
                use_weight_norm=self.use_weight_norm,
                name=f'{name}_encoder'
            )

        if self.use_decoder:
            self.decoder = Decoder1(self.filters, self.dilations, self.act_func, self.dense_size)
        self.project1 = tf.keras.layers.Dense(int(self.dense_size / 2), activation=None, name=f'{name}_dense1')

    def __call__(self, inputs, teacher=None):
        shape_obs = inputs['observations'].get_shape().as_list()
        if self.embedding_full == 'zeros':
            shape_fut = inputs['future'].get_shape().as_list()
            zeros = tf.zeros_like(inputs['future'])

            observations = tf.concat([zeros[:, shape_obs[1]:shape_fut[1], :shape_obs[2]],
                                      inputs['observations']], axis=1)
            future = inputs['future']
            calendar = self.embedings_calendar(inputs['calendar'])
            encoder_output = tf.concat([observations, future, calendar], axis=-1)
            if self.embedding_positional_flag:
                encoder_output = self.embedings_future(encoder_output)
                encoder_output += self.embedings_positional(encoder_output)
            #print(encoder_output.get_shape().as_list())
            if self.data_all:
                dec_input = encoder_output
                encoder_output, encoder_states = self.encoder(encoder_output)
            else:
                dec_input = encoder_output[:, shape_obs[1]-1:, :]
                encoder_output, encoder_states = self.encoder(encoder_output)
                encoder_output = encoder_output[:, shape_obs[1]-1:, :]
                encoder_states = [enc[:, shape_obs[1]-1:, :] for enc in encoder_states]
        elif self.embedding_full == 'discrete':
            observations = self.embedings_past([inputs['observations']])

            future = self.embedings_future(inputs['future'])

            calendar = self.embedings_calendar(inputs['calendar'])
            if self.embedding_positional_flag:
                observations += self.embedings_positional(observations)
                future += self.embedings_positional(future)
                calendar += self.embedings_positional(calendar)
            if self.data_all:
                dec_input = tf.concat([observations, future, calendar], axis=-1)
                observations, encoder_states_obs = self.encode_self(tf.concat([observations, calendar], axis=-1))
                future, encoder_states_fut = self.encode_future(tf.concat([future, calendar], axis=-1))
                encoder_states = [tf.concat([enc_obs, enc_fut], axis=-1)
                                  for enc_obs, enc_fut in zip(encoder_states_obs, encoder_states_fut)]
                encoder_output = tf.concat([observations, future], axis=-1)
            else:
                dec_input = tf.concat([observations[:, shape_obs[1]-1:, :], future[:, shape_obs[1]-1:, :],
                                       calendar[:, shape_obs[1]-1:, :]], axis=-1)
                observations, encoder_states_obs = self.encode_self(tf.concat([observations,
                                                           calendar], axis=-1))
                future, encoder_states_fut = self.encode_future(tf.concat([future,
                                                           calendar], axis=-1))
                encoder_states = [tf.concat([enc_obs, enc_fut], axis=-1)
                                  for enc_obs, enc_fut in zip(encoder_states_obs, encoder_states_fut)]
                encoder_output = tf.concat([observations, future], axis=-1)
                encoder_output = encoder_output[:, shape_obs[1]-1:, :]
                encoder_states = [enc[:, shape_obs[1]-1:, :] for enc in encoder_states]

        else:
            observations = self.embedings_past(inputs['observations'])
            future = self.embedings_future(inputs['future'])
            calendar = self.embedings_calendar(inputs['calendar'])
            encoder_output = tf.concat([observations, future, calendar], axis=-1)
            if self.embedding_positional_flag:
                encoder_output = self.embedings_future(encoder_output)
                encoder_output += self.embedings_positional(encoder_output)
            #print(encoder_output.get_shape().as_list())
            if self.data_all:
                dec_input = encoder_output
                encoder_output, encoder_states = self.encoder(encoder_output)
            else:
                dec_input = encoder_output[:, shape_obs[1]-1:, :]
                encoder_output, encoder_states = self.encoder(encoder_output)
                encoder_output = encoder_output[:, shape_obs[1]-1:, :]
                encoder_states = [enc[:, shape_obs[1]-1:, :] for enc in encoder_states]

        ###

        #print(f'encoder_output TF shape {encoder_output.get_shape().as_list()}')
        #print(f'encoder_states TF shape {len(encoder_states)}')

        # for i, state in enumerate(encoder_states):
            #print(f'encoder state {i} shape {state.get_shape().as_list()}')
        if self.use_decoder:
            encoder_output = self.decoder(tf.concat([dec_input, encoder_output], axis=-1), encoder_states)
            # #print(f'decoder_output shape {encoder_output.get_shape().as_list()}')
        return self.project1(encoder_output)


class Encoder(object):
    def __init__(self, kernel_sizes, dilation_rates, filters, activation, causal=True,
                 padding="causal",
                 use_batch_norm=True,
                 use_layer_norm=True,
                 use_weight_norm=True,
                 name='encoder'):
        self.filters = filters
        self.activation = activation
        self.conv_times = []
        for i, (kernel_size, dilation) in enumerate(zip(kernel_sizes, dilation_rates)):
            self.conv_times.append(
                ConvTemp(2 * filters, kernel_size, strides=1,
                         dilation_rate=1,
                         activation=activation,
                         padding=padding,
                         causal=causal,
                         use_batch_norm=use_batch_norm,
                         use_layer_norm=use_layer_norm,
                         use_weight_norm=use_weight_norm,
                         kernel_initializer="he_normal",
                         name=f'{name}_conv_{i}')
            )
        self.dense_time1 = DenseTemp(hidden_size=filters, activation="tanh", name=f"{name}_encoder_dense_time1")
        self.dense_time2 = DenseTemp(hidden_size=filters + filters, name=f"{name}_encoder_dense_time2")
        self.dense_time3 = DenseTemp(hidden_size=int(filters / len(dilation_rates)), activation=self.activation,
                                     name=f"{name}_encoder_dense_time3")
        self.final_activation = tf.keras.layers.Activation(self.activation)

    def __call__(self, x):
        #print(f'encoder input x TF shape {x.get_shape().as_list()}')
        inputs = self.dense_time1(inputs=x)  # batch_size * time_sequence_length * filters
        #print(f'encoder input initial TF shape {inputs.get_shape().as_list()}')
        skip_outputs = []
        conv_inputs = []
        i = 0
        for conv_time in self.conv_times:
            #print(f'encoder input step {i} TF shape {inputs.get_shape().as_list()}')
            dilated_conv = conv_time(inputs)
            #print(f'dilated_conv step {i} TF shape {dilated_conv.get_shape().as_list()}')
            conv_filter, conv_gate = tf.split(dilated_conv, 2, axis=2)
            #print(f'conv_filter step {i} TF shape {conv_filter.get_shape().as_list()}')
            #print(f'conv_gate step {i} TF shape {conv_gate.get_shape().as_list()}')
            dilated_conv = tf.nn.tanh(conv_filter) * tf.nn.sigmoid(conv_gate)
            #print(f'Activated dilated_conv step {i} TF shape {dilated_conv.get_shape().as_list()}')
            outputs = self.dense_time2(inputs=dilated_conv)
            #print(f'Dense 2 outputs {i} TF shape {outputs.get_shape().as_list()}')
            skips, residuals = tf.split(outputs, [self.filters, self.filters], axis=2)
            #print(f'skips step {i} TF shape {skips.get_shape().as_list()}')
            #print(f'residuals step {i} TF shape {residuals.get_shape().as_list()}')
            inputs += residuals
            conv_inputs.append(inputs)  # batch_size * time_sequence_length * filters
            skip_outputs.append(skips)
            i += 1

        skip_outputs = self.final_activation(tf.concat(skip_outputs, axis=2))
        #print(f'skip_outputs TF shape {skip_outputs.get_shape().as_list()}')
        h = self.dense_time3(skip_outputs)
        #print(f'output h TF shape {h.get_shape().as_list()}')
        return h, conv_inputs


class Decoder1(object):
    def __init__(
            self, filters, dilation_rates, activation, dense_hidden_size, name='decoder1'
    ) -> None:
        self.dilation_rates = dilation_rates
        self.dense1 = tf.keras.layers.Dense(filters, activation="tanh", name=f'{name}_dense1')
        self.dense2 = tf.keras.layers.Dense(2 * filters, use_bias=True, name=f'{name}_dense2')
        self.dense3 = tf.keras.layers.Dense(2 * filters, use_bias=False, name=f'{name}_dense3')
        self.dense4 = tf.keras.layers.Dense(2 * filters, name=f'{name}_dense4')
        self.dense5 = tf.keras.layers.Dense(dense_hidden_size, activation=activation, name=f'{name}_dense5')
        self.dense6 = tf.keras.layers.Dense((12 * len(dilation_rates)), name=f'{name}_dense_out')
        self.flat = tf.keras.layers.Flatten()
        self.final_activation = tf.keras.layers.Activation(activation)

    def __call__(
            self,
            decoder_features,
            encoder_outputs,
            **kwargs
    ):
        #print(f'decoder input TF shape {decoder_features.get_shape().as_list()}')
        this_input = decoder_features

        x = self.dense1(this_input)
        skip_outputs = []
        #print(f'decoder input initial TF shape {x.get_shape().as_list()}')

        for i, dilation in enumerate(self.dilation_rates):
            #print(f'dilation rate {dilation}')
            #print(f'decoder encoder_outputs step {i} TF shape {encoder_outputs[i].get_shape().as_list()}')
            state = encoder_outputs[i]
            #print(f'decoder state initial step {i} TF shape {state.get_shape().as_list()}')
            dense_state = self.dense2(state)
            dense_x = self.dense3(x)
            #print(f'decoder dense_state step {i} TF shape {dense_state.get_shape().as_list()}')
            #print(f'decoder dense_x step {i} TF shape {dense_x.get_shape().as_list()}')
            dilated_conv = dense_state + dense_x
            #print(f'decoder dilated_conv step {i} TF shape {dilated_conv.get_shape().as_list()}')
            conv_filter, conv_gate = tf.split(dilated_conv, 2, axis=-1)
            #print(f'conv_filter step {i} TF shape {conv_filter.get_shape().as_list()}')
            #print(f'conv_gate step {i} TF shape {conv_gate.get_shape().as_list()}')
            dilated_conv = tf.nn.tanh(conv_filter) * tf.nn.sigmoid(conv_gate)
            #print(f'decoder activated dilated_conv TF shape {dilated_conv.get_shape().as_list()}')
            out = self.dense4(dilated_conv)
            #print(f'decoder out TF shape {out.get_shape().as_list()}')
            skip, residual = tf.split(out, 2, axis=-1)
            #print(f'decoder skips step {i} TF shape {skip.get_shape().as_list()}')
            #print(f'decoder residuals step {i} TF shape {residual.get_shape().as_list()}')
            x += residual
            #print(f'decoder encoder_outputs final step {i} TF shape {encoder_outputs[i].get_shape().as_list()}')
            skip_outputs.append(skip)

        skip_outputs = self.final_activation(tf.concat(skip_outputs, axis=1))
        #print(f'decoder skip_outputs TF shape {skip_outputs.get_shape().as_list()}')
        skip_outputs = tf.transpose(skip_outputs, [0, 2, 1])
        skip_outputs = self.dense6(skip_outputs)
        skip_outputs = tf.transpose(skip_outputs, [0, 2, 1])
        skip_outputs = self.flat(skip_outputs)
        #print(f'decoder skip_outputs flat TF shape {skip_outputs.get_shape().as_list()}')
        skip_outputs = self.dense5(skip_outputs)
        #print(f'decoder skip_outputs dense 1 TF shape {skip_outputs.get_shape().as_list()}')
        return skip_outputs
