import tensorflow as tf


class DenseTemp(tf.keras.layers.Layer):
    def __init__(
            self,
            hidden_size,
            activation=None,
            kernel_initializer="glorot_uniform",
            kernel_regularizer=None,
            kernel_constraint=None,
            use_bias=True,
            bias_initializer="zeros",
            trainable=True,
            name=None,
    ):
        super(DenseTemp, self).__init__(trainable=trainable, name=name)
        self.hidden_size = hidden_size
        self.activation = tf.keras.activations.get(activation)
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        self.use_bias = use_bias
        self.bias_initializer = bias_initializer
        self.name_dense = name

    def build(self, input_shape):
        inputs_units = int(input_shape[-1])  # input.get_shape().as_list()[-1]
        self.kernel = self.add_weight(
            f"{self.name_dense}_kernel",
            shape=[inputs_units, self.hidden_size],
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            dtype=tf.float32,
            trainable=True,
        )
        if self.use_bias:
            self.bias = self.add_weight(
                f"{self.name_dense}_bias", shape=[self.hidden_size], initializer=self.bias_initializer,
                dtype=self.dtype, trainable=True
            )
        super(DenseTemp, self).build(input_shape)

    def call(self, inputs):
        output = tf.einsum("ijk,kl->ijl", inputs, self.kernel)

        if self.use_bias:
            output += self.bias

        if self.activation is not None:
            output = self.activation(output)
        return output

    def get_config(self):
        config = {
            "hidden_size": self.hidden_size,
        }
        base_config = super(DenseTemp, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class FeedForwardNetwork(tf.keras.layers.Layer):
    def __init__(self, hidden_size, output_size, act_func, dropout=0.0, name='FFN'):
        super(FeedForwardNetwork, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.act_func = act_func
        self.dropout = dropout
        self.name_ffn = name

    def build(self, input_shape):
        self.filter_dense_layer = tf.keras.layers.Dense(self.hidden_size, use_bias=True,
                                                        activation=self.act_func, name=f'{self.name_ffn}_dense1')
        self.output_dense_layer = tf.keras.layers.Dense(self.output_size, use_bias=True, name=f'{self.name_ffn}_dense2')
        self.drop = tf.keras.layers.Dropout(self.dropout)
        super(FeedForwardNetwork, self).build(input_shape)

    def call(self, x):
        """Feed Forward Network

        Parameters
        ----------
        x : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """
        output = self.filter_dense_layer(x)
        output = self.drop(output)
        output = self.output_dense_layer(output)
        return output

    def get_config(self):
        config = {
            "hidden_size": self.hidden_size,
            "filter_size": self.filter_size,
            "relu_dropout": self.relu_dropout,
        }
        base_config = super(FeedForwardNetwork, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
