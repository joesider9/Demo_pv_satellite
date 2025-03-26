import tensorflow as tf

from eforecast.deep_models.tf_2x.transformers.layers.attention_layer import CrossAttention
from eforecast.deep_models.tf_2x.transformers.layers.attention_layer import CausalSelfAttention
from eforecast.deep_models.tf_2x.transformers.layers.attention_layer import ProbAttention
from eforecast.deep_models.tf_2x.transformers.layers.attention_layer import FullAttention

from eforecast.deep_models.tf_2x.transformers.layers.embed_layer import RnnEmbedding
from eforecast.deep_models.tf_2x.transformers.layers.embed_layer import TimeEmbedding


class RNN(object):
    """RNN model"""

    def __init__(self, shape, params, name='RNN'):
        self.params = params
        self.num_dense_layer = params["num_stacked_layers"]
        self.num_heads = params["num_heads"]
        self.dropout = params["dropout"]
        if params["use_attention"]:
            self.attention = FullAttention(int((shape['observations'][0] + shape['observations'][1]) * self.num_heads),
                                           self.num_heads, self.dropout,
                                           name=f'{name}_full_attention')
            # self.ln_layer = tf.keras.layers.LayerNormalization()
        else:
            self.attention = None
            # self.ln_layer = None
        self.embedings_past = RnnEmbedding(shape['future'][0], return_state=params["embedding_state"],
                                           name=f'{name}_RnnEmbedding')
        self.embedings_calendar = TimeEmbedding(shape['calendar'], name=f'{name}_TimeEmbedding')
        self.encoder = Encoder(int(params["rnn_size"] * shape['future'][0]),
                               rnn_dropout=params['dropout'], name=f'{name}_encoder')

        self.denses = []
        self.dropouts = tf.keras.layers.Dropout(params['dropout'])
        for i in range(self.num_dense_layer):
            dense = tf.keras.layers.Dense(int(params["rnn_size"] * shape['future'][0] / (i + 1)), activation=params['act_func'],
                                          name=f'{name}_dense_{i}')
            self.denses.append(dense)
        self.flat = tf.keras.layers.Flatten()
        self.project1 = tf.keras.layers.Dense(params["dense_size"], activation=None, name=f'{name}_dense_out')

    def __call__(self, inputs, teacher=None):
        observations = self.embedings_past(inputs['observations'])
        future = inputs['future']
        calendar = self.embedings_calendar(inputs['calendar'])
        # print(f'observations has shape {observations.get_shape().as_list()}')
        # print(f'future has shape {future.get_shape().as_list()}')
        # print(f'calendar has shape {calendar.get_shape().as_list()}')

        if self.attention is not None:
            att = self.attention(observations, future, calendar)
            print(f'attention has shape {att.get_shape().as_list()}')
            observations = observations + att

        encoder_feature = tf.concat([observations, future, calendar], axis=-1)

        # print(f'encoder_feature inputs has shape {encoder_feature.get_shape().as_list()}')
        encoder_output, encoder_state = self.encoder(encoder_feature)
        # print(f'encoder_output output has shape {encoder_output.get_shape().as_list()}')
        # print(f'encoder_state 0 output has shape {encoder_state[0].get_shape().as_list()}')
        # print(f'encoder_state 1 output has shape {encoder_state[1].get_shape().as_list()}')
        encoder_output = tf.concat([encoder_output, tf.expand_dims(encoder_state[0], axis=1),
                                    tf.expand_dims(encoder_state[1], axis=1)], axis=1)
        # print(f'encoder_output output has shape {encoder_output.get_shape().as_list()}')
        encoder_output = self.dropouts(encoder_output)
        for i in range(self.num_dense_layer):
            encoder_output = self.denses[i](encoder_output)
            # print(f'denses {i} output has shape {encoder_output.get_shape().as_list()}')
            encoder_output = self.dropouts(encoder_output)

        encoder_output = self.flat(encoder_output)
        outputs = self.project1(encoder_output)
        # print(f'project1 output has shape {outputs.get_shape().as_list()}')
        return outputs


class Encoder(tf.keras.layers.Layer):
    def __init__(self, rnn_size, rnn_dropout, name='encoder', **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.rnn_size = rnn_size
        self.rnn_dropout = rnn_dropout
        self.name_encoder = name

    def build(self, input_shape):
        self.rnn = tf.keras.layers.LSTM(
            units=self.rnn_size,
            activation="tanh",
            return_state=True,
            return_sequences=True,
            dropout=self.rnn_dropout,
            name=f'{self.name_encoder}_LSTM2'
        )
        super(Encoder, self).build(input_shape)

    def call(self, inputs):
        output, state_memory, state_carry = self.rnn(inputs)
        state = (state_memory, state_carry)
        return output, state

    def get_config(self):
        config = {
            "rnn_size": self.rnn_size,
            "rnn_dropout": self.rnn_dropout,
            "dense_size": self.dense_size,
        }
        base_config = super(Encoder, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
