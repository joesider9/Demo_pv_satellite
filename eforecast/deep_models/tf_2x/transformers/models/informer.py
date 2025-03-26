import tensorflow as tf

from eforecast.deep_models.tf_2x.transformers.layers.attention_layer import FullAttention
from eforecast.deep_models.tf_2x.transformers.layers.attention_layer import ProbAttention
from eforecast.deep_models.tf_2x.transformers.layers.embed_layer import TokenEmbedding

from eforecast.deep_models.tf_2x.transformers.layers.embed_layer import RnnEmbedding
from eforecast.deep_models.tf_2x.transformers.layers.embed_layer import TimeEmbedding
from eforecast.deep_models.tf_2x.transformers.layers.mask_layer import CausalMask


class Informer(object):

    def __init__(self, shape, params, name='Informer'):
        self.params = params
        self.n_encoder_layers = params["n_encoder_layers"]
        self.n_decoder_layers = params["n_decoder_layers"]
        self.casual_mask_flag = params['casual_mask']
        self.dropout = params['dropout']
        self.num_heads = params['num_heads']
        self.attention_hidden_sizes = (self.num_heads *
                                       (shape['observations'][0] + 1 + shape['future'][0] + (2 * shape['calendar'][1])))
        self.dense_size = int(params['dense_size_rate'] * self.attention_hidden_sizes)
        self.act_func = params['act_func']
        self.act_func_dec = params['act_func_dec']
        self.use_conv = params['use_conv_encoder']
        self.embedding_full = params["embedding_full"]
        self.embedings_past = RnnEmbedding(shape['future'][0], return_state=True,
                                           name=f'{name}_RnnEmbedding')
        self.embedings_future = TokenEmbedding(shape['future'][0], name=f'{name}_TokenEmbedding')
        self.data_embedings_past = tf.keras.layers.Dense(shape['observations'][0] + 1, activation='elu', name=f'DataEmbedding')
        self.embedings_calendar = TimeEmbedding(shape['calendar'], name=f'{name}_TimeEmbedding')


        if self.embedding_full == 'discrete':
            attention_hidden_sizes_self = (self.num_heads *
             (shape['observations'][0] + 1 + (2 * shape['calendar'][1])))
            attn_layer_self = FullAttention(
                attention_hidden_sizes_self, self.num_heads, self.dropout,
                name=f'{name}_FullAttention_self'
            )
            self.encode_self = Encoder(
                layers=[
                    EncoderLayer(
                        attn_layer=attn_layer_self,
                        attention_hidden_sizes=attention_hidden_sizes_self,
                        ffn_dropout=self.dropout,
                        ffn_hidden_sizes=self.dense_size,
                        name=f'{name}_encode_selfLayer{i}'
                    )
                    for i in range(self.n_encoder_layers + 1)
                ],
                conv_layers=[
                    DistilConv(filters=attention_hidden_sizes_self // self.num_heads, name=f'{name}_self_DistilConv{i}')
                    for i in range(self.n_encoder_layers)
                ] if self.use_conv else None,
                norm_layer=None,
                name=f'{name}_encode_self'
            )
            attention_hidden_sizes_future = (self.num_heads *
                                       (shape['future'][0] + (2 * shape['calendar'][1])))
            attn_layer_future = FullAttention(
                attention_hidden_sizes_future, self.num_heads, self.dropout,
                name=f'{name}_FullAttention_future'
            )
            self.encode_future = Encoder(
                layers=[
                    EncoderLayer(
                        attn_layer=attn_layer_future,
                        attention_hidden_sizes=attention_hidden_sizes_future,
                        ffn_dropout=self.dropout,
                        ffn_hidden_sizes=self.dense_size,
                        name=f'{name}_EncoderfutureLayer{i}'
                    )
                    for i in range(self.n_encoder_layers + 1)
                ],
                conv_layers=[
                    DistilConv(filters=attention_hidden_sizes_future // self.num_heads, name=f'{name}_future_DistilConv{i}')
                    for i in range(self.n_encoder_layers)
                ] if self.use_conv else None,
                norm_layer=tf.keras.layers.LayerNormalization(name=f'{name}_future_DistilConv_norm'),
                name=f'{name}_future_encoder'
            )
        else:
            attn_layer = FullAttention(
                self.attention_hidden_sizes, self.num_heads, self.dropout,
                name=f'{name}_FullAttention'
            )
            self.encoder = Encoder(
                layers=[
                    EncoderLayer(
                        attn_layer=attn_layer,
                        attention_hidden_sizes=self.attention_hidden_sizes,
                        ffn_dropout=self.dropout,
                        ffn_hidden_sizes=self.dense_size,
                        name=f'{name}_EncoderLayer{i}'
                    )
                    for i in range(self.n_encoder_layers + 1)
                ],
                conv_layers=[
                    DistilConv(filters=self.attention_hidden_sizes // self.num_heads, name=f'{name}_DistilConv{i}')
                    for i in range(self.n_encoder_layers)
                ] if self.use_conv else None,
                norm_layer=None,
                name=f'{name}_encoder'
            )
        attention_hidden_sizes_dec = self.attention_hidden_sizes

        attn_layer1 = FullAttention(
            attention_hidden_sizes_dec, self.num_heads, self.dropout,
            name=f'{name}_FullAttention2'
        )


        attn_layer2 = FullAttention(attention_hidden_sizes_dec, self.num_heads, self.dropout,
                                    name=f'{name}_FullAttention3')
        self.decoder = Decoder(
            layers=[
                DecoderLayer(
                    attn_layer1=attn_layer1,
                    attn_layer2=attn_layer2,
                    attention_hidden_sizes=attention_hidden_sizes_dec,
                    ffn_dropout=self.dropout,
                    ffn_hidden_sizes=self.dense_size,
                    act_func=self.act_func_dec,
                    name=f'{name}_DecoderLayer{i}'
                )
                for i in range(self.n_decoder_layers)
            ]
        )
        self.flat = tf.keras.layers.Flatten()
        self.projection = tf.keras.layers.Dense(int(self.dense_size / self.n_encoder_layers
                                                    if self.use_conv else int(2 * self.dense_size)),
                                                activation=self.act_func, name=f'{name}_projection')

    def __call__(self, inputs, teacher=None):
        shape_obs = inputs['observations'].get_shape().as_list()
        if self.embedding_full == 'zeros':
            shape_fut = inputs['future'].get_shape().as_list()
            observations = self.data_embedings_past(inputs['observations'])
            shape_obs2 = observations.get_shape().as_list()
            future = self.embedings_future(inputs['future'])

            zeros = tf.zeros_like(future)


            calendar = self.embedings_calendar(inputs['calendar'])
            observations = tf.concat([zeros[:, shape_obs[1]:shape_fut[1], :shape_obs2[2]],
                                      observations], axis=1)

            encoder_output = tf.concat([observations, future, calendar], axis=-1)

            #print(encoder_output.get_shape().as_list())

            dec_input = encoder_output[:, shape_obs[1]:, :]
            encoder_output = self.encoder(encoder_output)
        elif self.embedding_full == 'discrete':
            observations = self.embedings_past(inputs['observations'])

            future = self.embedings_future(inputs['future'])

            calendar = self.embedings_calendar(inputs['calendar'])

            dec_input = tf.concat([observations[:, shape_obs[1]:, :], future[:, shape_obs[1]:, :],
                                   calendar[:, shape_obs[1]:, :]], axis=-1)
            observations = self.encode_self(tf.concat([observations, calendar], axis=-1))
            future = self.encode_future(tf.concat([future, calendar], axis=-1))

            encoder_output = tf.concat([observations, future], axis=-1)

        else:
            observations = self.embedings_past(inputs['observations'])
            future = self.embedings_future(inputs['future'])
            calendar = self.embedings_calendar(inputs['calendar'])
            encoder_output = tf.concat([observations, future, calendar], axis=-1)

            #print(encoder_output.get_shape().as_list())

            dec_input = encoder_output[:, shape_obs[1]:, :]
            encoder_output = self.encoder(encoder_output)
        #print(f'Encoder output after embedding{encoder_output.get_shape().as_list()}')

        casual_mask = None
        decoder_outputs = self.decoder(dec_input, encoder_output, x_mask=casual_mask)

        decoder_outputs = self.flat(decoder_outputs)
        decoder_outputs = self.projection(decoder_outputs)
        return decoder_outputs


class Encoder(tf.keras.layers.Layer):
    def __init__(self, layers, conv_layers=None, norm_layer=None, name='Encoder') -> None:
        super(Encoder, self).__init__()
        self.layers = layers
        self.conv_layers = conv_layers if conv_layers is not None else None
        self.norm_layer = norm_layer
        self.name_encoder = name

    def call(self, x, mask=None):
        """Informer encoder call function"""
        if self.conv_layers is not None:
            i = 0
            for attn_layer, conv_layer in zip(self.layers, self.conv_layers):
                x = attn_layer(x, mask)
                #print(f'Attention out of encoder {i} {x.get_shape().as_list()}')
                x = conv_layer(x)
                #print(f'distil conv_layer out  of encoder {i} {x.get_shape().as_list()}')
                i += 1
            x = self.layers[-1](x, mask)
            # #print(f'last Attention out  {x.get_shape().as_list()}')
        else:
            i = 0
            for attn_layer in self.layers:
                x = attn_layer(x, mask)
                #print(f'Attention out  of encoder {i} {x.get_shape().as_list()}')
                i += 1

        # if self.norm_layer is not None:
        #     x = self.norm_layer(x)
        return x


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, attn_layer, attention_hidden_sizes, ffn_hidden_sizes, ffn_dropout,
                 name='EncoderLayer') -> None:
        super().__init__()
        self.attn_layer = attn_layer
        self.attention_hidden_sizes = attention_hidden_sizes
        self.ffn_hidden_sizes = ffn_hidden_sizes
        self.ffn_dropout = ffn_dropout
        self.name_encoder = name

    def build(self, input_shape):
        self.drop = tf.keras.layers.Dropout(self.ffn_dropout)
        # self.norm1 = tf.keras.layers.LayerNormalization(name=f'{self.name_encoder}_norm')
        self.conv1 = tf.keras.layers.Conv1D(filters=self.ffn_hidden_sizes, kernel_size=1,
                                            name=f'{self.name_encoder}_conv1')
        self.conv2 = tf.keras.layers.Conv1D(filters=input_shape[-1], kernel_size=1,
                                            name=f'{self.name_encoder}_conv2')
        # self.norm2 = tf.keras.layers.LayerNormalization(name=f'{self.name_encoder}_norm2')
        super(EncoderLayer, self).build(input_shape)

    def call(self, x, mask=None):
        """Informer encoder layer call function"""
        input = x
        #print(f'Input of EncoderLayer {self.name_encoder} shape {x.get_shape().as_list()}')
        x = self.attn_layer(x, x, x, mask)
        #print(f'Attention of EncoderLayer {self.name_encoder} shape {x.get_shape().as_list()}')
        x = self.drop(x)
        x = x + input
        #print(f'Addition 1 of EncoderLayer {self.name_encoder} shape {x.get_shape().as_list()}')
        y = x
        y = self.conv1(y)
        #print(f'Conv1 of EncoderLayer {self.name_encoder} shape {y.get_shape().as_list()}')
        y = self.drop(y)
        y = self.conv2(y)
        #print(f'Conv2 of EncoderLayer {self.name_encoder} shape {y.get_shape().as_list()}')
        y = self.drop(y)
        y = x + y
        #print(f'Addition 2 of EncoderLayer {self.name_encoder} shape {y.get_shape().as_list()}')
        # y = self.norm2(y)
        #print(f'Output of EncoderLayer {self.name_encoder} shape {y.get_shape().as_list()}')
        return y

    def get_config(self):
        config = {
            "attention_hidden_sizes": self.attention_hidden_sizes,
            "ffn_hidden_sizes": self.ffn_hidden_sizes,
            "ffn_dropout": self.ffn_dropout,
        }
        base_config = super(EncoderLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class DistilConv(tf.keras.layers.Layer):
    def __init__(self, filters, name='DistilConv') -> None:
        super().__init__()
        self.filters = filters
        self.name_conv = name

    def build(self, input_shape):
        self.conv = tf.keras.layers.Conv1D(filters=self.filters, kernel_size=3, padding="causal",
                                           name=f'{self.name_conv}_conv1')
        # self.norm = tf.keras.layers.BatchNormalization()
        self.activation = tf.keras.layers.Activation("elu")
        self.pool = tf.keras.layers.MaxPool1D(pool_size=3, strides=2, padding="same", name=f'{self.name_conv}_pool1')
        super().build(input_shape)

    def call(self, x):
        """Informer distil conv"""
        x = self.conv(x)
        # x = self.norm(x)
        x = self.activation(x)
        x = self.pool(x)
        return x


class Decoder(tf.keras.layers.Layer):
    def __init__(self, layers, name='decoder'):
        super().__init__()
        self.layers = layers
        self.name_decoder = name

    def call(self, x, memory=None, x_mask=None, memory_mask=None):
        """Informer decoder call function"""
        for layer in self.layers:
            x = layer(x, memory, x_mask, memory_mask)

        return x


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, attn_layer1, attn_layer2, attention_hidden_sizes, ffn_hidden_sizes, ffn_dropout,
                 act_func, name='DecoderLayer') -> None:
        super().__init__()
        self.attn1 = attn_layer1
        self.attn2 = attn_layer2
        self.attention_hidden_sizes = attention_hidden_sizes
        self.ffn_hidden_sizes = ffn_hidden_sizes
        self.ffn_dropout = ffn_dropout
        self.act_func = act_func
        self.name_decoder = name

    def build(self, input_shape):
        self.conv1 = tf.keras.layers.Conv1D(filters=self.ffn_hidden_sizes, kernel_size=1,
                                            name=f'{self.name_decoder}_conv1')
        self.conv2 = tf.keras.layers.Conv1D(filters=input_shape[-1], kernel_size=1,
                                            name=f'{self.name_decoder}_conv2')
        self.drop = tf.keras.layers.Dropout(self.ffn_dropout)
        # self.norm1 = tf.keras.layers.LayerNormalization(name=f'{self.name_decoder}_norm1')
        # self.norm2 = tf.keras.layers.LayerNormalization(name=f'{self.name_decoder}_norm2')
        # self.norm3 = tf.keras.layers.LayerNormalization(name=f'{self.name_decoder}_norm3')
        self.activation = tf.keras.layers.Activation(self.act_func)
        super(DecoderLayer, self).build(input_shape)

    def call(self, x, memory=None, x_mask=None, memory_mask=None):
        """Informer decoder layer call function"""
        x0 = x
        #print(f'Input of DecoderLayer {self.name_decoder} shape {x.get_shape().as_list()}')
        x = self.attn1(x, x, x, mask=x_mask)
        #print(f'Attention 1 of DecoderLayer {self.name_decoder} shape {x.get_shape().as_list()}')
        x = self.drop(x)
        x = x + x0
        #print(f'Addition 1 of DecoderLayer {self.name_decoder} shape {x.get_shape().as_list()}')
        # x = self.norm1(x)

        x1 = x
        x = self.attn2(x, memory, memory, mask=memory_mask)
        #print(f'Attention 2 of DecoderLayer {self.name_decoder} shape {x.get_shape().as_list()}')
        x = self.drop(x)
        x = x + x1
        #print(f'Addition 2 of DecoderLayer {self.name_decoder} shape {x.get_shape().as_list()}')
        # x = self.norm2(x)

        x2 = x
        x = self.conv1(x)
        #print(f'Conv 1 of DecoderLayer {self.name_decoder} shape {x.get_shape().as_list()}')
        x = self.activation(x)
        x = self.drop(x)
        x = self.conv2(x)
        #print(f'Conv 2 of DecoderLayer {self.name_decoder} shape {x.get_shape().as_list()}')
        x = x + x2
        #print(f'Addition 3 of DecoderLayer {self.name_decoder} shape {x.get_shape().as_list()}')
        return x

    def get_config(self):
        config = {
            "attention_hidden_sizes": self.attention_hidden_sizes,
            "ffn_hidden_sizes": self.ffn_hidden_sizes,
            "ffn_dropout": self.ffn_dropout,
        }
        base_config = super(DecoderLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
