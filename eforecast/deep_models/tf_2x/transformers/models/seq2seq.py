import numpy as np
import pandas as pd
import tensorflow as tf
from eforecast.deep_models.tf_2x.transformers.layers.attention_layer import SelfAttention
from eforecast.deep_models.tf_2x.transformers.layers.attention_layer import FullAttention
from eforecast.deep_models.tf_2x.transformers.layers.attention_layer import CausalSelfAttention
from eforecast.deep_models.tf_2x.transformers.layers.attention_layer import CrossAttention
from eforecast.deep_models.tf_2x.transformers.layers.dense_layer import FeedForwardNetwork
from eforecast.deep_models.tf_2x.transformers.layers.embed_layer import RnnEmbedding
from eforecast.deep_models.tf_2x.transformers.layers.embed_layer import DataEmbedding
from eforecast.deep_models.tf_2x.transformers.layers.embed_layer import TokenEmbedding
from eforecast.deep_models.tf_2x.transformers.layers.embed_layer import TimeEmbedding
from eforecast.deep_models.tf_2x.transformers.layers.embed_layer import PositionalEmbedding
from eforecast.deep_models.tf_2x.transformers.layers.mask_layer import CausalMask


class Seq2Seq(object):
    def __init__(self, shape, params, name='Seq2Seq'):
        self.params = params
        self.embedding_full = params["embedding_full"]
        self.embedding_positional_flag = params['embedding_positional']
        self.embedings_past = RnnEmbedding(shape['future'][0], return_state=params["embedding_state"],
                                           name=f'{name}_RnnEmbedding')
        self.data_embedings_past = tf.keras.layers.Dense(4 * shape['future'][1], activation='elu', name=f'DataEmbedding')
        self.embedings_future = TokenEmbedding(4 * shape['future'][1], name=f'{name}_TokenEmbedding')
        self.embedings_calendar = TimeEmbedding(shape['calendar'], name=f'{name}_TimeEmbedding')
        self.embedings_positional = PositionalEmbedding()
        #print(f'Seq2seq with rnn size {params["rnn_size"]}')
        if self.embedding_full == 'discrete':
            self.encode_self = Encoder(params["rnn_size"], use_dense_enc_out=params['use_dense_enc_out'],
                                   use_dense_enc_state=params['use_dense_enc_state'], act_func='elu', name='encode_self')
            self.encode_future = Encoder(params["rnn_size"], use_dense_enc_out=params['use_dense_enc_out'],
                                   use_dense_enc_state=params['use_dense_enc_state'], act_func='elu', name='encode_future')
        else:
            self.encoder = Encoder(params["rnn_size"], use_dense_enc_out=params['use_dense_enc_out'],
                               use_dense_enc_state=params['use_dense_enc_state'], act_func='elu', name='encoder')
        if self.embedding_full == 'discrete':
            dec_dense_size = 2 * shape['future'][0]
            dec_output_size = 4 * shape['future'][1]
        else:
            dec_dense_size = shape['future'][0]
            dec_output_size = 4 * shape['future'][1]



        self.decoder = Decoder(params["rnn_size"], use_attention=params["use_attention"],
                               dense_size=dec_dense_size,
                               dense_size_out=dec_output_size,
                               dec_for_each_time=params["dec_for_each_time"],
                               attention_tf=params["attention_tf"],
                               num_heads=params["num_heads"], name='decoder')
        self.flatten = tf.keras.layers.Flatten()
        # self.project = tf.keras.layers.Dense(dec_output_size,
        #                                      activation=params['act_func'], name=f'{name}_dense_out')

    def __call__(self, inputs):
        observations = inputs['observations']
        observations = self.data_embedings_past(observations)
        shape_obs = observations.get_shape().as_list()

        dec_obs = observations[:, 0, :]
        if self.embedding_full == 'zeros':
            shape_fut = inputs['future'].get_shape().as_list()
            future = self.embedings_future(inputs['future'])

            zeros = tf.zeros_like(future)
            observations = tf.concat([zeros[:, shape_obs[1]:shape_fut[1], :shape_obs[2]],
                                      observations], axis=1)

            calendar = self.embedings_calendar(inputs['calendar'])
            encoder_output = tf.concat([observations, future, calendar], axis=-1)
            if self.embedding_positional_flag:
                encoder_output = self.embedings_future(encoder_output)
                encoder_output += self.embedings_positional(encoder_output)
            #print(encoder_output.get_shape().as_list())

            dec_input = encoder_output[:, shape_obs[1]-1:, :]
            encoder_output, encoder_states = self.encoder(encoder_output)
        elif self.embedding_full == 'discrete':
            observations = self.embedings_past(observations)

            future = self.embedings_future(inputs['future'])

            calendar = self.embedings_calendar(inputs['calendar'])
            if self.embedding_positional_flag:
                observations += self.embedings_positional(observations)
                future += self.embedings_positional(future)
                calendar += self.embedings_positional(calendar)

            dec_input = tf.concat([observations[:, shape_obs[1]-1:, :], future[:, shape_obs[1]-1:, :],
                                   calendar[:, shape_obs[1]-1:, :]], axis=-1)
            observations, encoder_states_obs = self.encode_self(tf.concat([observations,
                                                                           calendar], axis=-1))
            future, encoder_states_fut = self.encode_future(tf.concat([future,
                                                                       calendar], axis=-1))
            encoder_states = [tf.concat([enc_obs, enc_fut], axis=-1)
                                  for enc_obs, enc_fut in zip(encoder_states_obs, encoder_states_fut)]
            encoder_output = tf.concat([observations, future], axis=-1)

        else:
            observations = self.embedings_past(observations)
            future = self.embedings_future(inputs['future'])
            calendar = self.embedings_calendar(inputs['calendar'])
            encoder_output = tf.concat([observations, future, calendar], axis=-1)
            if self.embedding_positional_flag:
                encoder_output = self.embedings_future(encoder_output)
                encoder_output += self.embedings_positional(encoder_output)
            #print(encoder_output.get_shape().as_list())

            dec_input = encoder_output[:, shape_obs[1]-1:, :]
            encoder_output, encoder_states = self.encoder(encoder_output)

        #print(encoder_output.get_shape().as_list())

        #print(f'observations has shape {observations.get_shape().as_list()}')
        decoder_outputs = self.decoder(dec_input, dec_obs, encoder_states, encoder_output)
        decoder_outputs = self.flatten(decoder_outputs)

        return decoder_outputs


class Encoder(tf.keras.layers.Layer):
    def __init__(self, units, use_dense_enc_out=False, use_dense_enc_state=False,
                 act_func='elu', name='encoder', **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.dense_out = None
        self.dense_state = None
        self.dense = None
        self.rnn = None
        self.rnn_size = units
        self.use_dense_enc_state = use_dense_enc_state
        self.use_dense_enc_out = use_dense_enc_out
        self.act_func = act_func
        self.name_encoder = name

    def build(self, input_shape):
        #print(f'Encoder input shape in build {input_shape}')
        self.rnn = tf.keras.layers.LSTM(
            units=self.rnn_size * input_shape[-2],
            activation="tanh",
            return_state=True,
            return_sequences=True,
            name=f'{self.name_encoder}_lstm'
        )
        if self.use_dense_enc_out:
            self.dense_out = tf.keras.layers.Dense(self.rnn_size * input_shape[-2], activation=self.act_func,
                                                   name=f'{self.name_encoder}_dense_out')
        if self.use_dense_enc_state:
            self.dense_state1 = tf.keras.layers.Dense(self.rnn_size * input_shape[-2], activation=self.act_func,
                                                      name=f'{self.name_encoder}_dense_state1')
            self.dense_state2 = tf.keras.layers.Dense(self.rnn_size * input_shape[-2], activation=self.act_func,
                                                      name=f'{self.name_encoder}_dense_state2')

        super(Encoder, self).build(input_shape)

    def call(self, inputs):
        outputs, state1, state2 = self.rnn(inputs)
        #print(f'Encoder rnn outputs has shape {outputs.get_shape().as_list()}')
        #print(f'Encoder rnn state has shape {state1.get_shape().as_list()}')
        #print(f'Encoder rnn state has shape {state2.get_shape().as_list()}')
        if self.use_dense_enc_out:
            outputs = self.dense_out(outputs)
            #print(f'If use_dense_enc_out Encoder outputs has shape {outputs.get_shape().as_list()}')
        if self.use_dense_enc_state:
            state1 = self.dense_state1(state1)
            state2 = self.dense_state2(state2)
            #print(f'If use_dense_enc_state Encoder state1 has shape {state1.get_shape().as_list()}')
            #print(f'If use_dense_enc_state Encoder state2 has shape {state2.get_shape().as_list()}')
        return outputs, [state1, state2]


class Decoder(tf.keras.layers.Layer):
    def __init__(
            self,
            units,
            use_attention=False,
            dense_size=1,
            dense_size_out=1,
            dec_for_each_time=False,
            attention_tf=True,
            num_heads=1,
            name='decoder'
    ):
        super(Decoder, self).__init__()
        self.use_attention = use_attention
        self.rnn_size = units
        self.dense_size = dense_size
        self.dense_size_out = dense_size_out
        self.dec_for_each_time = dec_for_each_time
        self.attention_tf = attention_tf
        self.num_heads = num_heads
        self.name_decode = name

    def build(self, input_shape):
        #print(f'Decoder input shape in build {input_shape}')
        if self.dec_for_each_time:
            self.rnn_cell = []
            self.dense = []
            self.dense_att = []
            self.attention = []
            for i in range(input_shape[-2]):
                lstm = tf.keras.layers.LSTMCell(units=int(self.rnn_size * self.dense_size),
                                                name=f'{self.name_decode}_LSTM_{i}')
                dense_att = tf.keras.layers.Dense(units=int(self.rnn_size * self.dense_size), activation=None,
                                                  name=f'{self.name_decode}_Dense_att_{i}')
                dense = tf.keras.layers.Dense(units=self.dense_size_out, activation=None, name=f'{self.name_decode}_Dense_{i}')
                self.rnn_cell.append(lstm)
                self.dense.append(dense)
                self.dense_att.append(dense_att)
                if self.use_attention:
                    if not self.attention_tf:
                        attention = FullAttention(
                            hidden_size=int(input_shape[-1] * self.num_heads),
                            num_heads=self.num_heads,
                            name=f'{self.name_decode}_attention_{i}'
                        )
                    else:
                        attention = CrossAttention(num_heads=self.num_heads, key_dim=1,
                                                   name=f'{self.name_decode}_attention_{i}')
                    self.attention.append(attention)
        else:
            self.rnn_cell = tf.keras.layers.LSTMCell(units=int(self.rnn_size * self.dense_size),
                                                     name=f'{self.name_decode}_LSTM')
            self.dense_att = tf.keras.layers.Dense(units=int(self.rnn_size * self.dense_size), activation=None,
                                                   name=f'{self.name_decode}_Dense_att')
            self.dense = tf.keras.layers.Dense(units=self.dense_size_out, activation=None, name=f'{self.name_decode}_Dense')
            if self.use_attention:
                if not self.attention_tf:
                    self.attention = FullAttention(
                        hidden_size=int(input_shape[-1] * self.num_heads),
                        num_heads=self.num_heads,
                        name=f'{self.name_decode}_attention'
                    )
                else:
                    self.attention = CrossAttention(num_heads=self.num_heads, key_dim=1,
                                                    name=f'{self.name_decode}_attention')
        super().build(input_shape)

    def call(
            self,
            decoder_features,
            decoder_init_input,
            init_state,
            encoder_output,
            **kwargs
    ):
        #print(f'Input decoder_features (calendar) has shape {decoder_features.get_shape().as_list()}')
        #print(f'Input decoder_init_input (last observation) has shape {decoder_init_input.get_shape().as_list()}')
        #print(f'Input init_state (encoder_state) has shape {init_state[0].get_shape().as_list()}')
        #print(f'Input init_state (encoder_state) has shape {init_state[1].get_shape().as_list()}')
        #print(f'Input encoder_output has shape {encoder_output.get_shape().as_list()}')
        decoder_outputs = []
        prev_output = decoder_init_input
        prev_state = init_state
        time_length = decoder_features.get_shape().as_list()[-2]
        for i in range(time_length - 1, -1, -1):
            #print(f'For {i} time in decoder')
            this_input = prev_output
            #print(f'Initial this_input has shape {this_input.get_shape().as_list()}')
            this_input = tf.concat([this_input, decoder_features[:, i, :]], axis=-1)
            #print(
                # f'After concat with {i}th decoder_features (calendar) this_input has shape {this_input.get_shape().as_list()}')
            if self.dec_for_each_time:
                if self.use_attention:
                    if self.attention_tf:
                        att = self.attention[i](
                            tf.expand_dims(tf.concat(prev_state, 1), 1),
                            encoder_output,
                        )
                    else:
                        att = self.attention[i](
                            tf.expand_dims(tf.concat(prev_state, 1), 1),
                            k=encoder_output,
                            v=encoder_output,
                        )
                    att = tf.squeeze(att, 1)  # (batch, feature)
                    #print(
                        # f'Attention {i} has shape {att.get_shape().as_list()}')
                    this_input = self.dense_att[i](tf.concat([this_input, att], axis=-1))
                    #print(
                        # f'After concat with {i}th att this_input has shape {this_input.get_shape().as_list()}')
                this_output, this_state = self.rnn_cell[i](this_input, prev_state)
                this_state1, this_state2 = this_state
                #print(
                    # f'In {i}th this_output of rnn_cell has shape {this_output.get_shape().as_list()}')
                #print(
                    # f'In {i}th this_state1 of rnn_cell has shape {this_state1.get_shape().as_list()}')
                #print(f'In {i}th this_state2 of rnn_cell has shape {this_state2.get_shape().as_list()}')
                prev_output = self.dense[i](this_output)
                #print(
                    # f'In {i}th prev_output of {i}th dense has shape {prev_output.get_shape().as_list()}')
            else:
                if self.use_attention:
                    if self.attention_tf:
                        att = self.attention(
                            tf.expand_dims(tf.concat(prev_state, 1), 1),
                            encoder_output,
                        )
                    else:
                        att = self.attention(
                            tf.expand_dims(tf.concat(prev_state, 1), 1),
                            k=encoder_output,
                            v=encoder_output,
                        )
                    att = tf.squeeze(att, 1)  # (batch, feature)
                    #print(
                        # f'Attention has shape {att.get_shape().as_list()}')
                    this_input = self.dense_att(tf.concat([this_input, att], axis=-1))
                    #print(
                        # f'After concat with att this_input has shape {this_input.get_shape().as_list()}')
                this_output, this_state = self.rnn_cell(this_input, prev_state)
                this_state1, this_state2 = this_state
                #print(
                    # f'In this_output of rnn_cell has shape {this_output.get_shape().as_list()}')
                #print(
                    # f'In this_state1 of rnn_cell has shape {this_state1.get_shape().as_list()}')
                # #print(f'In this_state2 of rnn_cell has shape {this_state2.get_shape().as_list()}')
                prev_output = self.dense(this_output)
                #print(
                    # f'In prev_output of dense has shape {prev_output.get_shape().as_list()}')
            prev_state = [this_state1, this_state2]
            decoder_outputs.append(prev_output)

        decoder_outputs = tf.concat(decoder_outputs, axis=-1)
        return tf.expand_dims(decoder_outputs, -1)
