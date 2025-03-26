import tensorflow as tf

from eforecast.deep_models.tf_2x.transformers.layers.attention_layer import FullAttention, SelfAttention


class ConvTemp(tf.keras.layers.Layer):
    """Temporal convolutional layer"""

    def __init__(
            self,
            filters,
            kernel_size,
            strides=1,
            dilation_rate=1,
            activation="relu",
            padding="causal",
            causal=True,
            use_batch_norm=True,
            use_layer_norm=True,
            use_weight_norm=True,
            kernel_initializer="he_normal",
            name=None,
    ):
        super(ConvTemp, self).__init__(name=name)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.dilation_rate = dilation_rate
        self.activation = tf.keras.activations.get(activation)
        self.causal = causal
        self.padding = padding
        self.use_batch_norm = use_batch_norm
        self.use_layer_norm = use_layer_norm
        self.use_weight_norm = use_weight_norm
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.conv_name = name

    def build(self, input_shape) -> None:
        self.conv = tf.keras.layers.Conv1D(
            kernel_size=self.kernel_size,
            kernel_initializer=self.kernel_initializer,
            filters=self.filters,
            padding=self.padding,
            dilation_rate=self.dilation_rate,
            activation=self.activation,
            name=f'{self.conv_name}_conv'
        )
        self.norm = None
        # if self.use_weight_norm:
        #     from tensorflow_addons.layers import WeightNormalization
        #     # wrap it. WeightNormalization API is different than BatchNormalization or LayerNormalization.
        #     self.conv = WeightNormalization(self.conv, name=f'{self.conv_name}_batch')
        # elif self.use_batch_norm:
        #     self.norm = tf.keras.layers.BatchNormalization(name=f'{self.conv_name}_batch')
        # elif self.use_layer_norm:
        #     self.norm = tf.keras.layers.LayerNormalization(name=f'{self.conv_name}_batch')
        super(ConvTemp, self).build(input_shape)

    def call(self, inputs):
        if self.causal:
            padding_size = (self.kernel_size - 1) * self.dilation_rate
            # padding: dim 1 is batch, [0,0]; dim 2 is time, [padding_size, 0]; dim 3 is feature [0,0]
            inputs = tf.pad(inputs, [[0, 0], [padding_size, 0], [0, 0]])

        outputs = self.conv(inputs)
        if self.use_layer_norm or self.use_batch_norm:
            outputs = self.norm(outputs)
        return outputs

    def get_config(self):
        config = {
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            "strides": self.strides,
            "dilation_rate": self.dilation_rate,
            "casual": self.causal,
        }
        base_config = super(ConvTemp, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
