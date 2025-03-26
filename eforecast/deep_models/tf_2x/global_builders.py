import copy

import numpy as np
import pandas as pd
import tensorflow as tf

from eforecast.common_utils.train_utils import distance

from eforecast.deep_models.tf_2x.trainer import compute_tensors
from eforecast.deep_models.tf_2x.trainer import evaluate_activations

from eforecast.common_utils.train_utils import linear_output


def check_rbf_bounds(model, N, X_train, y_train, X_val, y_val, X_test, y_test,
                     params, best_clusters):
    warm = 0
    if best_clusters is None:
        raise ValueError('best_clusters is not computed')
    act_train = compute_tensors(model, 'activations', np.arange(N), X_train)
    act_val = compute_tensors(model, 'activations', np.arange(y_val.shape[0]), X_val)
    act_test = compute_tensors(model, 'activations', np.arange(y_test.shape[0]), X_test)
    mae_val_lin, mae_test_lin, sse_val_lin, sse_test_lin = linear_output(X_train, X_val, X_test,
                                                                         y_train, y_val, y_test,
                                                                         act_train, act_val, act_test,
                                                                         params)
    sum_act, min_act, max_act, mean_act, id_min, id_max = evaluate_activations(model,
                                                                               np.arange(N),
                                                                               X_train,
                                                                               params['thres_act'])
    min_samples = params['min_samples']
    max_samples = int(params['max_samples_ratio'] * y_train.shape[0])
    if min_act < min_samples:
        for variable in model.trainable_variables:
            if f'centroid_{id_min}' in variable.name or f'RBF_variance_{id_min}' in variable.name:
                variable.assign(best_clusters[variable.name])
                warm = 3
    if max_act > max_samples:
        for variable in model.trainable_variables:
            if f'centroid_{id_max}' in variable.name or f'RBF_variance_{id_max}' in variable.name:
                variable.assign(best_clusters[variable.name])
                warm = 3

    return mae_val_lin, mae_test_lin, sse_val_lin, sse_test_lin, sum_act, min_act, max_act, mean_act, warm


def assign_rbf(model, best_clusters):
    with tf.GradientTape() as tape:
        for variable in model.trainable_variables:
            if 'centroid' in variable.name or f'RBF_variance' in variable.name:
                variable.assign(best_clusters[variable.name])


def get_rbf(model):
    with tf.GradientTape() as tape:
        best_clusters = dict()
        for variable in model.trainable_variables:
            if 'centroid' in variable.name or 'RBF_variance' in variable.name:
                best_clusters[variable.name] = variable.numpy()
    return best_clusters


def get_stratify_batches_by_act(model, N, X_train, thres_act, batch_size, n_batch):
    act = compute_tensors(model, 'activations', np.arange(N), X_train)
    act[act >= thres_act] = 1
    act[act < thres_act] = 0
    prob = act.sum(axis=0) / act.shape[0]
    probs = prob[act.argmax(axis=1)]
    batches = [np.random.choice(N, batch_size, replace=False, p=probs / probs.sum())
               for _ in range(n_batch + 1)]
    return batches


class RBFLayer(tf.keras.layers.Layer):
    def __init__(self, thres_act, centroids, var_init):
        super(RBFLayer, self).__init__()
        self.var = []
        self.centroids = []
        self.thres_act = thres_act
        self.n_rules = centroids.shape[0]
        self.n_var = centroids.shape[1]
        self.var_init = []
        self.centroids_init = []
        for n in range(self.n_rules):
            self.var_init.append(var_init.iloc[n].values.reshape(1, -1))
            self.centroids_init.append(centroids[n].reshape(1, -1))

    def build(self, input_shape):
        for n in range(self.n_rules):
            centroids_init = tf.keras.initializers.constant(self.centroids_init[n])
            self.centroids.append(self.add_weight(f'centroid_{n}',
                                                  shape=[1, self.n_var],
                                                  dtype=tf.float32,
                                                  initializer=centroids_init,
                                                  trainable=False))
            var_init = tf.keras.initializers.constant(self.var_init[n])
            self.var.append(self.add_weight(f'RBF_variance_{n}',
                                            shape=[1, self.n_var],
                                            dtype=tf.float32,
                                            initializer=var_init,
                                            trainable=True))
        super(RBFLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        s = tf.shape(inputs)
        phi = []
        for n in range(self.n_rules):
            d1 = inputs - tf.tile(self.centroids[n], [s[0], 1])
            d = tf.sqrt(tf.reduce_sum(tf.pow(tf.divide(d1, tf.tile(self.var[n],
                                                                   [s[0], 1])),
                                             2), axis=1))
            phi.append(tf.expand_dims(tf.exp(tf.multiply(tf.constant(-1, dtype=tf.float32), tf.square(d))), axis=1))

        return phi


def build_fuzzy(fuzzy_inp, params):
    fuzzy_layer = RBFLayer(params['thres_act'], params['centroids'], params['var_init'])
    activations = fuzzy_layer(fuzzy_inp)
    return activations, fuzzy_layer


class apply_activations(tf.keras.layers.Layer):
    def __init__(self, thres_act):
        super(apply_activations, self).__init__()
        self.thres_act = thres_act

    def call(self, inputs, **kwargs):
        model_output = inputs[0]
        act_all = inputs[1]
        thres_act_tf = tf.constant(self.thres_act, tf.float32, name='thres_act_tf')
        thres_act_tf_up = tf.constant(self.thres_act + self.thres_act / 10, tf.float32, name='thres_act_tf_up')
        thres_act_tf_ml = tf.constant(10 / self.thres_act, tf.float32, name='thres_act_tf_ml')
        output_shape = model_output.get_shape().as_list()[-1]
        n_rules = act_all.get_shape().as_list()[-1]

        act_all_clip = tf.multiply(thres_act_tf_ml, tf.subtract(tf.clip_by_value(act_all, thres_act_tf,
                                                                                 thres_act_tf_up),
                                                                thres_act_tf))
        act_sum = tf.reduce_sum(act_all_clip, axis=1)
        act_sum_clipped = tf.clip_by_value(act_sum, 0.00000000001, n_rules + 1)
        act_all_weighted = tf.divide(act_all_clip, tf.tile(tf.expand_dims(act_sum_clipped, axis=1), [1, n_rules]))
        cluster_output_size = int(output_shape / n_rules)
        a_norm = tf.reshape(tf.tile(tf.expand_dims(act_all_weighted, -1), [1, 1, cluster_output_size]),
                            [-1, output_shape])
        model_output = tf.multiply(a_norm, model_output)
        return model_output


class act_nan_layer(tf.keras.layers.Layer):
    def __init__(self, thres_act):
        super(act_nan_layer, self).__init__()
        self.thres_act = thres_act

    def call(self, inputs, **kwargs):
        act_all = inputs
        thres_act_tf = tf.constant(self.thres_act, tf.float32, name='thres_act_tf')
        thres_act_tf_up = tf.constant(self.thres_act + self.thres_act / 10, tf.float32, name='thres_act_tf_up')
        thres_act_tf_ml = tf.constant(10 / self.thres_act, tf.float32, name='thres_act_tf_ml')
        n_rules = act_all.get_shape().as_list()[-1]

        act_all_clip = tf.multiply(thres_act_tf_ml, tf.subtract(tf.clip_by_value(act_all, thres_act_tf,
                                                                                 thres_act_tf_up),
                                                                thres_act_tf))
        act_sum = tf.reduce_sum(act_all_clip, axis=1)
        act_sum_clipped = tf.clip_by_value(act_sum, 0.00000000001, n_rules + 1)
        act_nan_err = tf.multiply(tf.subtract(act_sum_clipped, act_sum), tf.constant(1e11, tf.float32),
                                  name='act_nan_err')
        act_nan_err = tf.reduce_sum(act_nan_err, name='act_nan_err')

        return act_nan_err


def gauss_mf(x, mean, sigma):
    """
    Gaussian fuzzy membership function.

    Parameters
    ----------
    x : 1d tensor or iterable
        Independent variable.
    mean : float tensor constant
        Gaussian parameter for center (mean) value.
    sigma : float tensor constant
        Gaussian parameter for standard deviation.

    Returns
    -------
    y : 1d tensor
        Gaussian membership function for x.
    """

    return tf.exp(tf.divide(tf.multiply(tf.constant(-1, dtype=tf.float32), tf.square(tf.subtract(x, mean))),
                            tf.multiply(tf.constant(2, dtype=tf.float32), tf.square(sigma))))


def gbell_mf(x, a, b, c):
    """
        Generalized Bell function fuzzy membership generator.

    Parameters
    ----------
    x : 1d array
        Independent variable.
    a : float
        Bell function parameter controlling width. See Note for definition.
    b : float
        Bell function parameter controlling slope. See Note for definition.
    c : float
        Bell function parameter defining the center. See Note for definition.

    Returns
    -------
    y : 1d array
        Generalized Bell fuzzy membership function.

    Notes
    -----
    Definition of Generalized Bell function is:

        y(x) = 1 / (1 + abs([x - c] / a) ** [2 * b])
    """

    div = tf.abs(tf.divide(tf.subtract(x, c), a))
    p = tf.pow(div, 2 * b)
    value = tf.add(tf.constant(1, dtype=tf.float32), p)
    return tf.divide(tf.constant(1, dtype=tf.float32), value)


class FuzzyLayer(tf.keras.layers.Layer):
    def __init__(self, rules, thres_act):
        super(FuzzyLayer, self).__init__()
        self.fuzzy_vars = None
        self.rules = rules
        self.thres_act = thres_act

    def build(self, n):
        self.fuzzy_vars = dict()
        for rule_name, rule in self.rules.items():
            for mf in rule:
                self.fuzzy_vars['var_' + mf['name']] = dict()
                if mf['type'] == 'gauss':
                    var_init = tf.keras.initializers.constant(mf['param'][0])
                    self.fuzzy_vars['var_' + mf['name']]['mean'] = self.add_weight('var_' + mf['name'],
                                                                                   shape=[1],
                                                                                   dtype=tf.float32,
                                                                                   initializer=var_init,
                                                                                   trainable=True)

                    var_init = tf.keras.initializers.constant(mf['param'][1])
                    self.fuzzy_vars['var_' + mf['name']]['sigma'] = self.add_weight('var_' + mf['name'],
                                                                                    shape=[1],
                                                                                    dtype=tf.float32,
                                                                                    initializer=var_init,
                                                                                    trainable=True)

                else:
                    var_init = tf.keras.initializers.constant(mf['param'][0])
                    self.fuzzy_vars['var_' + mf['name']]['a'] = self.add_weight('var_' + mf['name'],
                                                                                shape=[1],
                                                                                dtype=tf.float32,
                                                                                initializer=var_init,
                                                                                trainable=True)

                    var_init = tf.keras.initializers.constant(mf['param'][1])
                    self.fuzzy_vars['var_' + mf['name']]['b'] = self.add_weight('var_' + mf['name'],
                                                                                shape=[1],
                                                                                dtype=tf.float32,
                                                                                initializer=var_init,
                                                                                trainable=True)
                    var_init = tf.keras.initializers.constant(mf['param'][2])
                    self.fuzzy_vars['var_' + mf['name']]['c'] = self.add_weight('var_' + mf['name'],
                                                                                shape=[1],
                                                                                dtype=tf.float32,
                                                                                initializer=var_init,
                                                                                trainable=True)
        super(FuzzyLayer, self).build(n)

    def call(self, fuzzy_inp, **kwargs):
        activations = None
        for rule_name, rule in self.rules.items():
            act_rule = None
            for mf in rule:
                if mf['type'] == 'gauss':
                    act = gauss_mf(fuzzy_inp[mf['name']], self.fuzzy_vars['var_' + mf['name']]['mean'],
                                   self.fuzzy_vars['var_' + mf['name']]['sigma'])
                else:
                    act = gbell_mf(fuzzy_inp[mf['name']], self.fuzzy_vars['var_' + mf['name']]['a'],
                                   self.fuzzy_vars['var_' + mf['name']]['b'],
                                   self.fuzzy_vars['var_' + mf['name']]['c'])
                act_rule = act if act_rule is None else tf.concat([act_rule, act], axis=1)
            act_rule = tf.reduce_prod(act_rule, axis=1, keepdims=True, name='act_' + rule_name)
            activations = act_rule if activations is None else tf.concat([activations, act_rule], axis=1)

        return activations


def check_fuzzy_performance(net_model, N,
                            X_train, y_train,
                            X_val, y_val,
                            X_test, y_test,
                            params,
                            init_clusters, best_clusters,
                            device, mae_old_lin,
                            mae_max_lin, mae_min_lin,
                            sse_old_lin, sse_max_lin,
                            sse_min_lin, explode_clusters):
    mae_val_lin, mae_test_lin, sse_val_lin, sse_test_lin, \
        sum_act, min_act, max_act, mean_act, warm = \
        check_rbf_bounds(net_model, N,
                         X_train, y_train,
                         X_val, y_val,
                         X_test, y_test,
                         params, init_clusters)
    mae_lin = np.hstack([mae_val_lin, mae_test_lin])
    flag_mae_lin, mae_old_lin, mae_max_lin, mae_min_lin = distance(mae_lin, mae_old_lin,
                                                                   mae_max_lin, mae_min_lin)
    sse_lin = np.hstack([sse_val_lin, sse_test_lin])
    flag_sse_lin, sse_old_lin, sse_max_lin, sse_min_lin = distance(sse_lin, sse_old_lin,
                                                                   sse_max_lin, sse_min_lin)
    if (flag_mae_lin and flag_sse_lin) and not (mae_val_lin > 1 or mae_test_lin > 1):
        best_clusters = get_rbf(net_model)
    if (mae_val_lin > 1 or mae_test_lin > 1) \
            and explode_clusters:
        for param, weight in best_clusters.items():
            weight *= 1.25
        assign_rbf(net_model, best_clusters)
        mae_val_lin, mae_test_lin, sse_val_lin, sse_test_lin, \
            sum_act, min_act, max_act, mean_act, warm = \
            check_rbf_bounds(net_model, N,
                             X_train, y_train,
                             X_val, y_val,
                             X_test, y_test,
                             params, init_clusters)
        warm = 4
    return (net_model, best_clusters, sum_act, min_act, max_act, mean_act, warm, mae_old_lin, mae_max_lin,
            mae_min_lin, sse_old_lin, sse_max_lin, sse_min_lin, mae_val_lin, mae_test_lin)