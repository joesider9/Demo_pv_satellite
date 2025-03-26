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



class Transformer(object):

    def __init__(self, shape, params, name='Transformer'):
        self.params = params
        self.n_encoder_layers = params["n_encoder_layers"]
        self.embedding_full = params["embedding_full"]
        self.embedding_positional_flag = params['embedding_positional']
        self.casual_mask_flag = params['casual_mask']
        self.data_all = params['data_all']
        self.embedings_past = RnnEmbedding(shape['future'][0], return_state=params["embedding_state"],
                                           name=f'{name}_RnnEmbedding')
        self.embedings_future = TokenEmbedding(shape['future'][0], name=f'{name}_TokenEmbedding')
        self.embedings_calendar = TimeEmbedding(shape['calendar'], name=f'{name}_TimeEmbedding')
        self.embedings_positional = PositionalEmbedding()
        if self.embedding_full == 'discrete':
            self.encode_self = Encoder(self.n_encoder_layers, params['num_heads'], params['dropout'],
                                       params['dense_size'],
                                       params['act_func'])
            self.encode_future = Encoder(self.n_encoder_layers, params['num_heads'], params['dropout'],
                                         params['dense_size'],
                                         params['act_func'])
            self.encode_calendar = Encoder(self.n_encoder_layers, params['num_heads'], params['dropout'],
                                           params['dense_size'],
                                           params['act_func'])
        else:
            self.encoder = Encoder(self.n_encoder_layers, params['num_heads'], params['dropout'], params['dense_size'],
                                       params['act_func'], attention_tf=params["attention_tf"], residual=params['residual'])
        self.decoder = Decoder2(
            embed_layer=DataEmbedding(name=f'decoder_embed'),
            att_layers=[
                DecoderLayer2(
                    params["n_decoder_layers"],
                    params["num_heads"],
                    params["dropout"],
                    params["dense_size"],
                    params['act_func'],
                    attention_tf=params["attention_tf"],
                    residual=params["residual"],
                    name=f'{name}_decoder2_{i}'
                )
                for i in range(params["n_decoder_layers"])
            ],
            dense_size=int(params['dense_size'] / params['num_heads']),
            act_func = params['act_func'],
            dropout=params["dropout"],
            time_distributed=params['time_distributed']
        )
        self.project = tf.keras.layers.Dense(int(params['dense_size'] / 5),
                                             activation=params['act_func'], name=f'{name}_dense_out')

    def __call__(self, inputs, teacher=None):
        shape_obs = inputs['observations'].get_shape().as_list()
        if self.embedding_full == 'zeros':
            shape_fut = inputs['future'].get_shape().as_list()
            zeros = tf.zeros_like(inputs['future'])

            observations = tf.concat([zeros[:, shape_obs[1]:shape_fut[1], :shape_obs[2]],
                                      inputs['observations']], axis=1)
            future = inputs['future']
            calendar = inputs['calendar']
            encoder_output = tf.concat([observations, future, calendar], axis=-1)
            if self.embedding_positional_flag:
                encoder_output = self.embedings_future(encoder_output)
                encoder_output += self.embedings_positional(encoder_output)
            print(encoder_output.get_shape().as_list())
            if self.data_all:
                dec_input = observations
                encoder_output = self.encoder(encoder_output)
            else:
                dec_input = encoder_output[:, shape_obs[1]-1:, :]
                encoder_output = self.encoder(encoder_output)
        elif self.embedding_full == 'discrete':
            observations = self.embedings_past(inputs['observations'])

            future = self.embedings_future(inputs['future'])

            calendar = self.embedings_calendar(inputs['calendar'])
            if self.embedding_positional_flag:
                observations += self.embedings_positional(observations)
                future += self.embedings_positional(future)
                calendar += self.embedings_positional(calendar)
            if self.data_all:
                dec_input = observations
                observations = self.encode_self(observations)
                future = self.encode_future(future)
                calendar = self.encode_calendar(calendar)
                encoder_output = tf.concat([observations, future,
                                            calendar], axis=-1)
            else:
                dec_input = tf.concat([observations[:, shape_obs[1]-1:, :], future[:, shape_obs[1]-1:, :],
                                            calendar[:, shape_obs[1]-1:, :]], axis=-1)
                observations = self.encode_self(observations)
                future = self.encode_future(future)
                calendar = self.encode_calendar(calendar)
                encoder_output = tf.concat([observations, future,
                                            calendar], axis=-1)

        else:
            observations = self.embedings_past(inputs['observations'])
            future = inputs['future']
            calendar = self.embedings_calendar(inputs['calendar'])
            encoder_output = tf.concat([observations, future, calendar], axis=-1)
            if self.embedding_positional_flag:
                encoder_output = self.embedings_future(encoder_output)
                encoder_output += self.embedings_positional(encoder_output)
            print(encoder_output.get_shape().as_list())
            if self.data_all:
                dec_input = observations
                encoder_output = self.encoder(encoder_output)
            else:
                dec_input = encoder_output[:, shape_obs[1]-1:, :]
                encoder_output = self.encoder(encoder_output)
        if self.casual_mask_flag:
            B, L, _ = tf.shape(dec_input)
            casual_mask = CausalMask(B * self.params["num_heads"], L).mask
        else:
            casual_mask = None
        decoder_outputs = self.decoder(dec_input, encoder_output, x_mask=casual_mask)
        decoder_outputs = self.project(decoder_outputs)
        return decoder_outputs


class Encoder(tf.keras.layers.Layer):
    def __init__(
            self,
            n_encoder_layers,
            num_heads,
            dropout,
            dense_size,
            act_func,
            attention_tf=False,
            residual=False,
            name='encoder'
    ):
        super(Encoder, self).__init__()
        self.n_encoder_layers = n_encoder_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.dense_size = dense_size
        self.act_func = act_func
        self.attention_tf = attention_tf
        self.residual = residual
        self.layers = []
        self.name_encoder = name

    def build(self, input_shape) -> None:
        for i in range(self.n_encoder_layers):
            if self.attention_tf:
                attention_layer = CausalSelfAttention(num_heads=self.num_heads, key_dim=1,
                                                name=f'{self.name_encoder}_self_attention_{i}')
            else:
                attention_layer = SelfAttention(int(input_shape[-1] * self.num_heads),
                                                self.num_heads, self.dropout,
                                                name=f'{self.name_encoder}_self_attention_{i}')
            ffn_layer = FeedForwardNetwork(self.dense_size, input_shape[-1], self.act_func,
                                           dropout=self.dropout,
                                           name=f'{self.name_encoder}_FFN_{i}')
            ln_layer1 = tf.keras.layers.LayerNormalization()
            ln_layer2 = tf.keras.layers.LayerNormalization()
            self.layers.append([attention_layer, ln_layer1, ffn_layer, ln_layer2])
        super(Encoder, self).build(input_shape)

    def call(self, encoder_inputs, encoder_mask=None):
        x = encoder_inputs
        for _, layer in enumerate(self.layers):
            attention_layer, ln_layer1, ffn_layer, ln_layer2 = layer
            enc = x
            if self.attention_tf:
                enc = attention_layer(enc, False if encoder_mask is None else True)
            else:
                enc = attention_layer(enc, encoder_mask)
                enc = x + enc
            enc1 = ffn_layer(enc)
            if self.residual:
                x = enc + enc1
            else:
                x = enc1
        return x

    def get_config(self):
        config = {
            "n_encoder_layers": self.n_encoder_layers,
            "dense_size": self.attention_hidden_sizes,
            "num_heads": self.num_heads,
            "attention_dropout": self.attention_dropout,
            "ffn_dropout": self.ffn_dropout,
        }
        base_config = super(Encoder, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Decoder2(tf.keras.layers.Layer):
    def __init__(self, embed_layer=None, att_layers=None, dense_size=None, act_func='elu', dropout=0.0,
                 time_distributed=True, name='decoder2') -> None:
        super().__init__()
        self.att_layers = att_layers
        self.decoder_embedding = embed_layer

        if time_distributed:
            self.dense = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(int(dense_size / 2), activation=act_func,
                                                                               name=f'{name}_proj1'),
                                                      name=f'{name}_time_proj1')
            self.drop = tf.keras.layers.TimeDistributed(tf.keras.layers.Dropout(dropout))
        else:
            self.dense = tf.keras.layers.Dense(int(dense_size / 2), activation=act_func, name=f'{name}_proj1')
            self.drop = tf.keras.layers.Dropout(dropout)

    def call(self, x, memory, x_mask=None, memory_mask=None):
        """Transformer decoder2

        Parameters
        ----------
        x : _type_
            _description_
        memory : _type_
            _description_
        x_mask : _type_, optional
            _description_, by default None
        memory_mask : _type_, optional
            _description_, by default None

        Returns
        -------
        tf.Tensor
            _description_
        """
        x = self.decoder_embedding(x)
        for layer in self.att_layers:
            x = layer(x, memory, x_mask, memory_mask)
        x = self.dense(x)
        x = self.drop(x)
        return x


class DecoderLayer2(tf.keras.layers.Layer):
    def __init__(
            self,
            n_decoder_layers,
            num_heads,
            dropout,
            dense_size,
            act_func,
            eps=1e-7,
            attention_tf=False,
            residual=False,
            name='decoder_layer2'
    ):
        super(DecoderLayer2, self).__init__()
        self.n_decoder_layers = n_decoder_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.dense_size = dense_size
        self.act_func = act_func
        self.eps = eps
        self.attention_tf = attention_tf
        self.residual = residual
        self.layers = []
        self.name_decoder = name

    def build(self, input_shape):
        for i in range(self.n_decoder_layers):
            if self.attention_tf:
                self_attention_layer = CausalSelfAttention(num_heads=self.num_heads, key_dim=1,
                                                     name=f'{self.name_decoder}_self_attention_{i}')
                enc_dec_attention_layer = CrossAttention(num_heads=self.num_heads, key_dim=1,
                                                        name=f'{self.name_decoder}_full_attention_{i}')
            else:
                self_attention_layer = SelfAttention(int(input_shape[-1] * self.num_heads), self.num_heads, self.dropout,
                                                     name=f'{self.name_decoder}_self_attention_{i}')
                enc_dec_attention_layer = FullAttention(int(input_shape[-1] * self.num_heads), self.num_heads, self.dropout,
                                                        name=f'{self.name_decoder}_full_attention_{i}')
            feed_forward_layer = FeedForwardNetwork(self.dense_size, input_shape[-1], self.act_func,
                                                    dropout=self.dropout,
                                                    name=f'{self.name_decoder}_FFN_{i}')
            self.layers.append(
                [self_attention_layer, enc_dec_attention_layer, feed_forward_layer]
            )
        super(DecoderLayer2, self).build(input_shape)

    def call(self, decoder_inputs, encoder_memory, decoder_mask=None, memory_mask=None):
        x = decoder_inputs

        for _, layer in enumerate(self.layers):
            self_attention_layer, enc_dec_attention_layer, ffn_layer = layer
            dec = x
            if self.attention_tf:
                dec1 = self_attention_layer(dec)
                mask = True if memory_mask is not None else False
                dec2 = enc_dec_attention_layer(dec1, encoder_memory,
                                               mask=mask)
            else:
                dec = self_attention_layer(dec, mask=decoder_mask)
                dec1 = (x + dec)
                dec1 = enc_dec_attention_layer(dec1, encoder_memory, encoder_memory, mask=memory_mask)
                dec2 = (x + dec1)
            dec2 = ffn_layer(dec2)
            if self.residual:
                x = (dec1 + dec2)  # note that don't repeat ln
            else:
                x = dec2
            # x = dec1 + dec2
        return x

    def get_config(self):
        config = {
            "n_decoder_layers": self.n_decoder_layers,
            "dense_size": self.attention_hidden_sizes,
            "num_heads": self.num_heads,
            "attention_dropout": self.attention_dropout,
            "ffn_dropout": self.ffn_dropout,
        }
        base_config = super(DecoderLayer2, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
