import tensorflow as tf

from eforecast.deep_models.tf_2x.transformers.models.temporal_conv_net import TemporalConvNet
from eforecast.deep_models.tf_2x.transformers.models.rnn import RNN
from eforecast.deep_models.tf_2x.transformers.models.transformer import Transformer
from eforecast.deep_models.tf_2x.transformers.models.informer import Informer
from eforecast.deep_models.tf_2x.transformers.models.seq2seq import Seq2Seq
from eforecast.deep_models.tf_2x.transformers.models.tide import Tide


class AutoModel(object):
    def __init__(self, use_model, shape, params, name='transformer'):
        if use_model.lower() == "rnn":
            self.model = RNN(shape, params, name=name)
        elif use_model.lower() == "seq2seq":
            self.model = Seq2Seq(shape, params, name=name)
        elif use_model.lower() == "tcn":
            self.model = TemporalConvNet(shape, params, name=name)
        elif use_model.lower() == "transformer":
            self.model = Transformer(shape, params, name=name)
        elif use_model.lower() == "informer":
            self.model = Informer(shape, params, name=name)
        elif use_model.lower() == "tide":
            self.model = Tide(shape, params, name=name)
        else:
            raise ValueError("unsupported model of {} yet".format(use_model))

    def __call__(self, x):
        return self.model(x)

    def build_model(self, inputs):
        outputs = self.model(inputs)
        return tf.keras.Model([inputs], [outputs])  # to handles the Keras symbolic tensors for tf2.3.1

    def from_pretrained(self, name: str):
        return
