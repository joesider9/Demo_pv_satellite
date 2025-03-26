import tensorflow as tf
import sys


def get_rated(rated, y):
    if rated is not None:
        norm_val = tf.constant(1, tf.float32, name='rated')
    else:
        norm_val = y
    return norm_val


class LearningRateSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Learning rate schedule."""

    def __init__(self, learning_rate, hidden_size, warmup_steps, n_batch=100):
        """Constructor.

    Args:
      learning_rate: float scalar, the base learning rate.
      hidden_size: int scalar, the hidden size of continuous representation.
      warmup_steps: int scalar, the num of warm-up steps
    """
        super(LearningRateSchedule, self).__init__()
        self._learning_rate = tf.cast(learning_rate, 'float32')
        self._hidden_size = hidden_size
        self.n_batch = n_batch
        self._warmup_steps = tf.cast(warmup_steps * n_batch, 'float32')

    def __call__(self, global_step):
        """Computes learning rate with linear warmup and rsqrt decay.

    Args:
      global_step: int scalar tensor, the current global step.

    Returns:
      learning_rate: float scalar tensor, the learning rate as a function of
        the input `global_step`.
    """
        global_step = tf.cast(global_step, 'float32')
        learning_rate = self._learning_rate / tf.sqrt(tf.maximum(global_step, self._warmup_steps))
        return learning_rate


class LearningRateScheduleCos(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Learning rate schedule."""

    def __init__(self, learning_rate, warmup_steps, n_batch=100):
        """Constructor.

    Args:
      learning_rate: float scalar, the base learning rate.
      hidden_size: int scalar, the hidden size of continuous representation.
      warmup_steps: int scalar, the num of warm-up steps
    """
        super(LearningRateScheduleCos, self).__init__()
        self._learning_rate = tf.cast(learning_rate, 'float32')
        self.n_batch = tf.cast(n_batch, 'float32')
        self._warmup_steps = tf.cast(warmup_steps, 'float32') * n_batch

    def __call__(self, global_step):
        global_step = tf.cast(global_step, 'float32')
        # WarmUp
        progress = tf.maximum(tf.cast(0.1, 'float32'), (global_step - self._warmup_steps) / (20 * self.n_batch))
        learning_rate = (self._learning_rate *
                         tf.maximum(tf.cast(0.1, 'float32'),
                                    0.5 * (1.0 +
                                           tf.cos(3.14 * progress))))
        # tf.print("learning_rate:", learning_rate, output_stream=sys.stdout)
        # tf.print("global_step:", global_step, output_stream=sys.stdout)
        return learning_rate


def optimize_bulk(learning_rate, rated=None, probabilistic=False, quantiles=None, n_batch=200):
    learning_rate_ = LearningRateScheduleCos(learning_rate, 5, n_batch=n_batch)

    # if learning_rate < 1e-2:
    #     learning_rate_ = LearningRateScheduleCos(learning_rate, 20, n_batch=n_batch)
    # else:
    #     learning_rate_ = LearningRateSchedule(learning_rate, 49, 20, n_batch=n_batch)
    # Create losses
    if probabilistic:
        raise NotImplementedError('probabilistic')
        # losses = []
        # for i, q in enumerate(quantiles):
        #     error = y - model_output[i]
        #     loss = tf.reduce_mean(tf.maximum(q * error, (q - 1) * error),
        #                           axis=-1)
        #
        #     losses.append(loss)
        # err_out = tf.add_n(losses)
        # cost_out = tf.reduce_mean(err_out)
    else:
        loss_out = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM)
    train_out = tf.keras.optimizers.Adam(learning_rate=learning_rate_)
    if rated is not None:
        accuracy_out = tf.keras.metrics.MeanAbsoluteError()
        sse_out = tf.keras.metrics.MeanSquaredError()
    else:
        accuracy_out = tf.keras.metrics.MeanAbsolutePercentageError()
        sse_out = tf.keras.metrics.MeanSquaredError()
    return train_out, loss_out, accuracy_out, sse_out


def optimize(is_global=False, rated=None, learning_rate=1e-4, is_fuzzy=False,
             probabilistic=False, quantiles=None, n_batch=100, epochs=600):
    with tf.name_scope("optimizers") as scope:
        trainers, losses, MAEs, SSEs = optimize_bulk(learning_rate, rated=rated,
                                                     probabilistic=probabilistic,
                                                     quantiles=quantiles,
                                                     n_batch=n_batch)

    return trainers, losses, MAEs, SSEs, learning_rate
