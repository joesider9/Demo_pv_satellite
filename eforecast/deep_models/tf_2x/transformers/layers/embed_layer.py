# -*- coding: utf-8 -*-
# @author: Longxing Tan, tanlongxing888@163.com
"""Layer for :py:class:`~eforecast.deep_models.tf_2x.tfts.models.transformer`"""

from typing import Any, Callable, Dict, Optional, Tuple, Type, Union

import numpy as np
import tensorflow as tf


class TokenEmbedding(tf.keras.layers.Layer):
    """
    A layer that performs token embedding.

    Args:
        embed_size (int): The size of the embedding.

    Input shape:
        - 3D tensor with shape `(batch_size, time_steps, input_dim)`

    Output shape:
        - 3D tensor with shape `(batch_size, time_steps, embed_size)`
    """

    def __init__(self, embed_size: int, name='token_embedding'):
        super(TokenEmbedding, self).__init__()
        self.embed_size = embed_size
        self.name_embed = name

        self.tokenConv = tf.keras.layers.Conv1D(filters=embed_size,
                                                kernel_size=3, padding='causal',
                                                activation='linear', name=f'{name}_token_embedding')
        self.activation = tf.keras.layers.LeakyReLU()

    def call(self, x):
        """
        Performs the token embedding.

        Args:
            x (tensor): Input tensor.

        Returns:
            Tensor: Embedded tensor.
        """
        y = self.activation(self.tokenConv(x))
        return y

    def get_config(self):
        config = {"embed_size": self.embed_size}
        base_config = super(TokenEmbedding, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class TokenRnnEmbedding(tf.keras.layers.Layer):
    def __init__(self, embed_size: int, name='token_embedding') -> None:
        super().__init__()
        self.embed_size = embed_size
        self.name_embed = name

    def build(self, input_shape: Tuple[Optional[int], ...]):
        self.rnn = tf.keras.layers.GRU(self.embed_size, return_sequences=True, return_state=True, name=f'{self.name_embed}_GRU')
        super().build(input_shape)

    def call(self, x):
        """
        Forward pass of the layer.

        Parameters
        ----------
        x : tf.Tensor
            Input tensor of shape `(batch_size, sequence_length, input_dim)`.

        Returns
        -------
        y : tf.Tensor
            Output tensor of shape `(batch_size, sequence_length, embed_size)`.
        """
        y, _ = self.rnn(x)
        return y

    def get_config(self):
        config = {"embed_size": self.embed_size}
        base_config = super(TokenRnnEmbedding, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class RnnEmbedding(tf.keras.layers.Layer):
    def __init__(self, embed_size, return_state=False, name='rnn_embedding') -> None:
        super().__init__()
        self.embed_size = embed_size
        self.name_embed = name
        self.return_state = return_state

    def build(self, input_shape: Tuple[Optional[int], ...]):
        if self.return_state:
            self.rnn = tf.keras.layers.SimpleRNN(self.embed_size, return_state=True, return_sequences=True,
                                            name=f'{self.name_embed}_GRU')
        else:
            self.rnn = tf.keras.layers.LSTM(self.embed_size, name=f'{self.name_embed}_lstm')
        super().build(input_shape)

    def call(self, x):
        """
        Forward pass of the layer.

        Parameters
        ----------
        x : tf.Tensor
            Input tensor of shape `(batch_size, observation_sequence_length, input_dim)`.

        Returns
        -------
        y : tf.Tensor
            Output tensor of shape `(batch_size, observation_prediction_sequence_length, observation_sequence_length+1)`.
        """
        if not self.return_state:
            sequence_output = self.rnn(x)
            y = tf.expand_dims(sequence_output, axis=-1)
        else:
            sequence_output, final_state = self.rnn(x)
            y = tf.transpose(tf.concat([final_state[:, tf.newaxis, :], sequence_output], axis=1), [0, 2, 1])
        return y

    def get_config(self):
        config = {"embed_size": self.embed_size}
        base_config = super(RnnEmbedding, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, max_len=5000, name='positional_embedding'):
        super(PositionalEmbedding, self).__init__()
        self.max_len = max_len
        self.name_embed = name

    def build(self, input_shape):
        super(PositionalEmbedding, self).build(input_shape)

    def call(self, x, masking=True):
        """
        Applies positional encoding to the input tensor.

        Parameters:
        x (tf.Tensor): Input tensor of shape (batch_size, sequence_length, embedding_dim).
        masking (bool, optional): If True, applies masking to the output tensor, by default True.

        Returns:
        tf.Tensor: Output tensor of the same shape as the input tensor, after applying positional encoding.
        """
        E = x.get_shape().as_list()[-1]  # static
        batch_size, seq_length = tf.shape(x)[0], tf.shape(x)[1]  # dynamic

        position_ind = tf.tile(tf.expand_dims(tf.range(seq_length), 0), [batch_size, 1])  # => batch_size * seq_length
        position_enc = np.array(
            [[pos / np.power(10000, (i - i % 2) / E) for i in range(E)] for pos in range(self.max_len)]
        )

        position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])
        position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])
        position_enc = tf.convert_to_tensor(position_enc, tf.float32)  # (maxlen, E)

        outputs = tf.nn.embedding_lookup(position_enc, position_ind, name=f'{self.name_embed}_lookup')
        if masking:
            outputs = tf.where(tf.equal(x, 0), x, outputs)
        return tf.cast(outputs, tf.float32)

    def get_config(self):
        config = {"max_len": self.max_len}
        base_config = super(PositionalEmbedding, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, max_len: int = 5000, name='positional_encoding'):
        super(PositionalEncoding, self).__init__()
        self.max_len = max_len
        self.name_embed = name

    def build(self, input_shape: Tuple[Optional[int], ...]):
        super(PositionalEncoding, self).build(input_shape)

    def call(self, x, masking=True):
        """Applies positional encoding to the input tensor.

        Parameters
        ----------
        x : tf.Tensor
            The input tensor of shape (batch_size, seq_length, embed_dim).
        masking : bool, optional
            Whether to mask padded values, by default True.

        Returns
        -------
        tf.Tensor
            The output tensor of shape (batch_size, seq_length, embed_dim) with positional encoding applied.
        """
        E = x.get_shape().as_list()[-1]  # static
        batch_size, seq_length = tf.shape(x)[0], tf.shape(x)[1]  # dynamic
        with tf.name_scope("position_encode"):
            # # => batch_size * seq_length
            position_ind = tf.tile(tf.expand_dims(tf.range(seq_length), 0), [batch_size, 1])
            position_enc = np.array(
                [[pos / np.power(10000, (i - i % 2) / E) for i in range(E)] for pos in range(self.max_len)]
            )

            position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])
            position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])
            position_enc = tf.convert_to_tensor(position_enc, tf.float32)  # (maxlen, E)

            outputs = tf.nn.embedding_lookup(position_enc, position_ind, name=f'{self.name_embed}_lookup')
            if masking:
                outputs = tf.where(tf.equal(x, 0), x, outputs)
        return tf.cast(outputs, tf.float32)

    def get_config(self):
        config = {"max_len": self.max_len}
        base_config = super(PositionalEncoding, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class TimeEmbedding(tf.keras.layers.Layer):
    def __init__(self, shape, name='TimeEmbedding', **kwargs):
        super(TimeEmbedding, self).__init__()
        self.shape = shape
        self.name_emb = name

    def build(self, input_shape):
        self.weights_linear = self.add_weight(name=f'{self.name_emb}_weight_linear',
                                              shape=self.shape,
                                              initializer='uniform',
                                              trainable=True)

        self.bias_linear = self.add_weight(name=f'{self.name_emb}_bias_linear',
                                           shape=self.shape,
                                           initializer='uniform',
                                           trainable=True)

        self.weights_periodic = self.add_weight(name=f'{self.name_emb}_weight_periodic',
                                                shape=self.shape,
                                                initializer='uniform',
                                                trainable=True)

        self.bias_periodic = self.add_weight(name=f'{self.name_emb}_bias_periodic',
                                             shape=self.shape,
                                             initializer='uniform',
                                             trainable=True)

    def call(self, x):
        '''Calculate linear and periodic time features'''

        time_linear = self.weights_linear * x + self.bias_linear  # Linear time feature

        time_periodic = tf.math.sin(tf.multiply(x, self.weights_periodic) + self.bias_periodic)

        return tf.concat([time_linear, time_periodic], axis=-1)  # shape = (batch, seq_len, 2)

    def get_config(self):  # Needed for saving and loading model with custom layer
        config = super().get_config().copy()
        config.update({'seq_len': self.seq_len})
        return config


class DataEmbedding(tf.keras.layers.Layer):
    def __init__(self, dropout= 0.0, name='embedding'):
        """
        Data Embedding layer.

        Args:
            embed_size (int): Embedding size for tokens.
            dropout (float, optional): Dropout rate to apply. Defaults to 0.0.
        """
        self.name_emb = name
        self.dropout = dropout
        super(DataEmbedding, self).__init__()


    def build(self, input_shape: Tuple[Optional[int], ...]):
        self.value_embedding = TokenEmbedding(input_shape[-1], name=f'{self.name_emb}_token')
        self.positional_embedding = PositionalEncoding(name=f'{self.name_emb}_positional')
        self.dropout = tf.keras.layers.Dropout(self.dropout)
        super(DataEmbedding, self).build(input_shape)

    def call(self, x):
        """
        Forward pass of the layer.

        Args:
            x (tf.Tensor): Input tensor of shape (batch_size, seq_length).

        Returns:
            tf.Tensor: Output tensor of shape (batch_size, seq_length, embed_size).
        """
        ve = self.value_embedding(x)
        pe = self.positional_embedding(ve)
        return self.dropout(ve + pe)

    def get_config(self):
        config = {"embed_size": self.embed_size}
        base_config = super(DataEmbedding, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
