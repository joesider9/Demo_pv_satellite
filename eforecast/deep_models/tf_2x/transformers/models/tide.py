import tensorflow as tf

from eforecast.deep_models.tf_2x.transformers.layers.embed_layer import TimeEmbedding


class MLPResidual(tf.keras.layers.Layer):
    """Simple one hidden state residual network."""
    def __init__(self, hidden_dim, output_dim, layer_norm=False, dropout_rate=0.0, name='MLP_res'):
        super(MLPResidual, self).__init__()
        self.lin_a = tf.keras.layers.Dense(
            hidden_dim,
            activation='relu',
            name=f'{name}_dense_a'
        )
        self.lin_b = tf.keras.layers.Dense(
            output_dim,
            activation=None,
            name=f'{name}_dense_b'
        )
        self.lin_res = tf.keras.layers.Dense(
            output_dim,
            activation=None,
            name=f'{name}_dense_res'
        )
        if layer_norm:
          self.lnorm = tf.keras.layers.LayerNormalization(name=f'{name}_norm')
        self.layer_norm = layer_norm
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, inputs):
        """Call method."""
        h_state = self.lin_a(inputs)
        out = self.lin_b(h_state)
        out = self.dropout(out)
        res = self.lin_res(inputs)
        if self.layer_norm:
          return self.lnorm(out + res)
        return out + res


class _make_dnn_residual(tf.keras.layers.Layer):
    def __init__(self, hidden_dims, layer_norm=False, dropout_rate=0.0, name='dnn'):
        super(_make_dnn_residual, self).__init__()
        self.layers = []
        if len(hidden_dims) < 2:
            self.layers.append(tf.keras.layers.Dense(
                hidden_dims[-1],
                activation=None,
                name=f'{name}_dense_0'
            ))

        for i, hdim in enumerate(hidden_dims[:-1]):
            self.layers.append(
                MLPResidual(
                    hdim,
                    hidden_dims[i + 1],
                    layer_norm=layer_norm,
                    dropout_rate=dropout_rate,
                    name=f'{name}_mlp_res_{i}'
                )
            )

    def call(self, inputs):
        out = inputs
        for layer in self.layers:
            out = layer(out)
        return out

class Tide(object):
    def __init__(self, shape, params, name='Tide'):

        self.params = params
        self.layer_norm = params["layer_norm"]
        self.dropout_rate = params["dropout"]
        self.hidden_dims = [params['hidden_dims']] * params['num_layers']
        self.time_encoder_size = [params['time_encoder_size'], params['decoder_output_size']]
        self.decoder_output_size = params['decoder_output_size']
        self.final_decoder_hidden = params['final_decoder_hidden']
        self.output_size = params["output_size"]
        self.embedings_calendar_past = TimeEmbedding([shape['observations'][0], shape['calendar'][-1]], name=f'{name}_TimeEmbedding_past')
        self.embedings_calendar_future = TimeEmbedding([shape['future'][0] - shape['observations'][0],
                                                        shape['calendar'][-1]],
                                                       name=f'{name}_TimeEmbedding_future')
        self.flat_obs = tf.keras.layers.Flatten()
        self.flat_extra_past = tf.keras.layers.Flatten()
        self.flat_extra_future = tf.keras.layers.Flatten()
        self.flat_calendar_past = tf.keras.layers.Flatten()
        self.flat_calendar_future = tf.keras.layers.Flatten()
        self.encoder_past = _make_dnn_residual(
            self.hidden_dims,
            layer_norm=self.layer_norm,
            dropout_rate=self.dropout_rate,
            name = f'{name}_encoder_past'
        )
        self.encoder_future = _make_dnn_residual(
            self.hidden_dims,
            layer_norm=self.layer_norm,
            dropout_rate=self.dropout_rate,
            name=f'{name}_encoder_future'
        )
        self.decoder = _make_dnn_residual(
            self.hidden_dims[:-1] + [self.decoder_output_size],
            layer_norm=self.layer_norm,
            dropout_rate=self.dropout_rate,
            name=f'{name}_decoder'
        )
        self.linear_obs = tf.keras.layers.Dense(
            self.output_size,
            activation=None,
            name=f'{name}_linear_obs'
        )
        self.linear_extra = tf.keras.layers.Dense(
            self.output_size,
            activation=None,
            name=f'{name}_linear_extra'
        )
        self.time_encoder = _make_dnn_residual(
            self.time_encoder_size,
            layer_norm=self.layer_norm,
            dropout_rate=self.dropout_rate,
            name=f'{name}_time_encoder'
        )
        self.final_decoder = MLPResidual(
            hidden_dim=self.final_decoder_hidden,
            output_dim=self.output_size,
            layer_norm=self.layer_norm,
            dropout_rate=self.dropout_rate,
            name=f'{name}_final_decoder'
        )

    def __call__(self, inputs):
        observations = inputs['observations']
        shape = observations.get_shape().as_list()
        observations = self.flat_obs(observations)
        extra_past = self.flat_extra_past(inputs['future'][:, :shape[1], :])
        extra_future = self.flat_extra_future(inputs['future'][:, shape[1]:, :])
        calendar_past = self.flat_calendar_past(self.embedings_calendar_past(inputs['calendar'][:, :shape[1], :]))
        calendar_future = self.flat_calendar_future(self.embedings_calendar_future(inputs['calendar'][:, shape[1]:, :]))

        print(f'observations has shape {observations.get_shape().as_list()}')
        print(f'future has shape {extra_past.get_shape().as_list()}')
        print(f'future has shape {extra_future.get_shape().as_list()}')
        print(f'calendar has shape {calendar_past.get_shape().as_list()}')
        print(f'calendar has shape {calendar_future.get_shape().as_list()}')

        residual_out = self.linear_obs(observations)
        encoder_input_past = tf.concat([observations, extra_past, calendar_future], axis=1)
        encoding_past = self.encoder_past(encoder_input_past)
        encoder_input_future = tf.concat([extra_future, calendar_future], axis=1)
        encoding_future = self.encoder_future(encoder_input_future)
        decoding = tf.concat([encoding_past, encoding_future], axis=1)
        decoder_out = self.decoder(decoding)
        final_in = tf.concat([decoder_out, encoding_future], axis=1)
        out = self.final_decoder(final_in)  # B x H x 1
        out += residual_out
        return out
