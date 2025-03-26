import os
import sys
import copy
import psutil
import joblib
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans

try:
    import tensorflow as tf
except:
    import torch

import multiprocessing as mp

from time import sleep
from contextlib import contextmanager
import yagmail

from sklearn.linear_model import LinearRegression


class GpuQueue:

    def __init__(self, N_GPUS, all_gpus=False):
        self.queue = mp.Manager().Queue()
        if all_gpus:
            all_idxs = list(range(N_GPUS))
        else:
            all_idxs = [N_GPUS]
        for idx in all_idxs:
            self.queue.put(idx)

    @contextmanager
    def one_gpu_per_process(self):
        current_idx = self.queue.get()
        yield current_idx
        self.queue.put(current_idx)


def find_min_max_var(inputs, n_rules, n_var, centroids, var, thres_act):
    s = np.shape(inputs)
    phi = []
    for n in range(n_rules):
        d1 = inputs.values - np.tile(centroids[n], [s[0], 1])
        d = np.sqrt(np.sum(np.power(d1 / np.tile(var.values[n], [s[0], 1]), 2), axis=1))
        phi.append(np.expand_dims(np.exp(-1 * np.square(d)), axis=1))
    act_all_eval = np.concatenate(phi, axis=1)
    act_all_eval[act_all_eval >= thres_act] = 1
    act_all_eval[act_all_eval < thres_act] = 0
    return act_all_eval.sum(axis=0)


def check_VAR_if_all_nans(inputs, n_rules, n_var, centroids, var, thres_act):
    s = np.shape(inputs)
    dist = []
    phi = []
    for n in range(n_rules):
        d1 = inputs.values - np.tile(centroids[n], [s[0], 1])
        dist.append(np.expand_dims(np.sqrt(np.sum(np.square(d1), axis=1)), axis=1))
        d = np.sqrt(np.sum(np.power(d1 / np.tile(var.values[n], [s[0], 1]), 2), axis=1))
        phi.append(np.expand_dims(np.exp(-1 * np.square(d)), axis=1))
    dist = np.concatenate(dist, axis=1)
    activations = np.concatenate(phi, axis=1)
    indices = np.where(np.all(activations <= thres_act, axis=1))[0]
    len_nan = 0
    while indices.shape[0] > 0:
        d = dist[indices[0]]
        clust = np.argmin(d)
        var.values[clust] += thres_act
        dist = []
        phi = []
        for n in range(n_rules):
            d1 = inputs.values - np.tile(centroids[n], [s[0], 1])
            dist.append(np.expand_dims(np.sqrt(np.sum(np.square(d1), axis=1)), axis=1))
            d = np.sqrt(np.sum(np.power(d1 / np.tile(var.values[n], [s[0], 1]), 2), axis=1))
            phi.append(np.expand_dims(np.exp(-1 * np.square(d)), axis=1))
        dist = np.concatenate(dist, axis=1)
        activations = np.concatenate(phi, axis=1)
        indices = np.where(np.all(activations <= thres_act, axis=1))[0]
    return var


def create_centroids(X_train, y, params):
    if X_train is None:
        raise ValueError('X_train is not provided')
    dates = y.index.intersection(X_train.index)
    split = int(0.75 * len(dates))
    X_tr = X_train.loc[dates[:split]].values
    y_tr = y.loc[dates[:split]].values
    X_val = X_train.loc[dates[split:]].values
    y_val = y.loc[dates[split:]].values
    c_best = None
    inertia = np.inf
    from eforecast.clustering.rbf_ols_network import RBFols
    for w in np.linspace(0.8, 1.4, 6):
        c = RBFols(params['rated'], n_clusters=params['n_rules'], thres_act=params['thres_act'],
                   width=w * np.abs(X_tr).mean())
        c.fit(X_tr, y_tr, X_val, y_val)
        if c.err < inertia:
            c_best = copy.deepcopy(c)
            inertia = c.err
    if c_best is not None:
        min_samples = params['min_samples']
        max_samples = int(params['max_samples_ratio'] * X_train.shape[0])
        centroids = c_best.model['centroids'].astype(np.float32)
        widths = 1 / c_best.model['Radius'].astype(np.float32)

        var_init = pd.DataFrame(widths, index=['c' + str(i) for i in range(centroids.shape[0])],
                                columns=['v' + str(i) for i in range(centroids.shape[1])])
        var_init = check_VAR_if_all_nans(X_train, params['n_rules'], centroids.shape[1], centroids, var_init,
                                         params['thres_act'])
        n_samples = find_min_max_var(X_train, params['n_rules'], centroids.shape[1], centroids, var_init,
                                     params['thres_act'])
        ind_small = np.where(n_samples < min_samples)[0]
        ind_large = np.where(n_samples > max_samples)[0]
        while ind_small.shape[0] != 0:
            var_init.iloc[ind_small] += 0.001
            n_samples = find_min_max_var(X_train, params['n_rules'], centroids.shape[1], centroids, var_init,
                                         params['thres_act'])
            ind_small = np.where(n_samples < min_samples)[0]
        while ind_large.shape[0] != 0:
            var_init.iloc[ind_large] -= 0.001
            n_samples = find_min_max_var(X_train, params['n_rules'], centroids.shape[1], centroids, var_init,
                                         params['thres_act'])
            ind_large = np.where(n_samples > max_samples)[0]
        params['centroids'] = centroids
        params['var_init'] = var_init
        params['rules'] = [f'rule_{i}' for i in range(params['n_rules'])]

    return params


def fix_convolutional_names(horizon_type, experiment, conv_dim):
    for group, branch in experiment.items():
        for i, layer in enumerate(branch):
            if 'conv' in layer[0]:
                if conv_dim is not None:
                    if horizon_type == 'multi-output':
                        experiment[group][i] = (f'time_distr_conv_{conv_dim}d', layer[1])
                    else:
                        experiment[group][i] = (f'conv_{conv_dim}d', layer[1])
                else:
                    raise ValueError('Cannot find conv_dim parameter')
    return experiment


def feed_data(batch, x, y, data, target, lr_pl, lr):
    feed_dict = dict()
    if isinstance(x, dict):
        for key in x.keys():
            if isinstance(x[key], dict):
                for key_1 in x[key].keys():
                    feed_dict.update({x[key][key_1]: data[key][key_1][batch]})
            else:
                feed_dict.update({x[key]: data[key][batch]})
    else:
        feed_dict.update({x: data[batch]})
    feed_dict.update({y: target[batch]})
    feed_dict.update({lr_pl: lr})
    return feed_dict


def feed_data_eval(x, data):
    feed_dict = dict()
    if isinstance(x, dict):
        for key in x.keys():
            if isinstance(x[key], dict):
                for key_1 in x[key].keys():
                    feed_dict.update({x[key][key_1]: data[key][key_1]})
            else:
                feed_dict.update({x[key]: data[key]})
    else:
        feed_dict.update({x: data})
    return feed_dict


def distance(obj_new, obj_old, obj_max, obj_min, weights=None):
    if np.any(np.isinf(obj_old)):
        obj_old = obj_new.copy()
        obj_max = obj_new.copy()
        return True, obj_old, obj_max, obj_min
    if np.any(np.isinf(obj_min)) and not np.all(obj_max == obj_new):
        obj_min = obj_new.copy()
    d = 0
    for i in range(obj_new.shape[0]):
        if obj_max[i] < obj_new[i]:
            obj_max[i] = obj_new[i]
        if obj_min[i] > obj_new[i]:
            obj_min[i] = obj_new[i]
        if weights is None:
            if obj_max[i] - obj_min[i] < 1e-6:
                d += (obj_new[i] - obj_old[i])
            else:
                d += (obj_new[i] - obj_old[i]) / (obj_max[i] - obj_min[i])
        else:
            if obj_max[i] - obj_min[i] < 1e-6:
                d += weights[i] * ((obj_new[i] - obj_old[i]))
            else:
                d += weights[i] * ((obj_new[i] - obj_old[i]) / (obj_max[i] - obj_min[i]))

    if weights is not None:
        d = d / np.sum(weights)
    if d < 0:
        obj_old = obj_new.copy()
        return True, obj_old, obj_max, obj_min
    else:
        return False, obj_old, obj_max, obj_min


def split_validation_set(x):
    if isinstance(x, dict):
        for values in x.values():
            x1 = copy.deepcopy(values)
            break
    else:
        x1 = copy.deepcopy(x)
    if x1.shape[0] > 1000:
        partitions = 250
        ind_list = []
        ind = np.arange(x1.shape[0])
        for i in range(0, x1.shape[0], partitions):
            if (i + partitions + 1) > x1.shape[0]:
                ind_list.append(ind[i:])
            else:
                ind_list.append(ind[i:i + partitions])
    else:
        ind_list = [np.arange(x1.shape[0])]
    return ind_list


def calculate_cpus(n_cpus):
    warm = psutil.cpu_percent()
    average_load = np.mean(psutil.cpu_percent(interval=5, percpu=True)[:n_cpus])

    return n_cpus - int(n_cpus * average_load / 100)


def get_tf_config(n_jobs):
    n_cpus = calculate_cpus(n_jobs)
    if sys.platform != 'linux' and n_cpus > int(n_jobs / 3):
        config = tf.compat.v1.ConfigProto(allow_soft_placement=True,
                                          intra_op_parallelism_threads=2,
                                          inter_op_parallelism_threads=2)
        config.gpu_options.allow_growth = True
    else:
        config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
    return config


def lr_schedule(epoch, lr=1e-04):
    if epoch < 50:
        # WarmUp
        return np.linspace(lr / 10, lr, 50)[epoch]
    else:
        lr_step = 0.5 * lr * (1 + np.cos(np.pi * (epoch - 50) / float(20)))
        return np.maximum(lr / 10, lr_step)


def check_if_all_nans(activations, thres_act, return_len_nan=False):
    indices = np.where(np.all(activations.values < thres_act, axis=1))[0]
    len_nan = 0
    if indices.shape[0] > 0:
        len_nan = indices.shape[0]
        for ind in indices:
            act = activations.loc[ind]
            clust = act.idxmax()
            activations.loc[ind, clust] = thres_act
    if return_len_nan:
        return activations, len_nan
    else:
        return activations


def linear_output(X_train, X_val, X_test, y_train, y_val, y_test, act_train, act_val, act_test, params):
    rules = params['rules']
    act_train = pd.DataFrame(act_train, columns=rules)
    act_train = check_if_all_nans(act_train, params['thres_act'])
    act_val = pd.DataFrame(act_val, columns=rules)
    act_val, len_nan_val = check_if_all_nans(act_val, params['thres_act'], return_len_nan=True)
    act_test = pd.DataFrame(act_test, columns=rules)
    act_test, len_nan_test = check_if_all_nans(act_test, params['thres_act'], return_len_nan=True)
    for key1 in X_train.keys():
        if isinstance(X_train[key1], dict):
            for key2 in X_train.keys():
                if isinstance(X_train[key1][key2], pd.DataFrame):
                    x_ = np.copy(X_train[key1][key2].values)
                    x_val_ = np.copy(X_val[key1][key2].values)
                    x_test_ = np.copy(X_test[key1][key2].values)
                else:
                    x_ = np.copy(X_train[key1][key2])
                    x_val_ = np.copy(X_val[key1][key2])
                    x_test_ = np.copy(X_test[key1][key2])
        else:
            if isinstance(X_train[key1], pd.DataFrame):
                x_ = np.copy(X_train[key1].values)
                x_val_ = np.copy(X_val[key1].values)
                x_test_ = np.copy(X_test[key1].values)
            else:
                x_ = np.copy(X_train[key1])
                x_val_ = np.copy(X_val[key1])
                x_test_ = np.copy(X_test[key1])


    if len(x_.shape) > 2:
        shape = x_.shape
        x_ = x_.reshape(-1, np.prod(shape[1:]))
        x_val_ = x_val_.reshape(-1, np.prod(shape[1:]))
        x_test_ = x_test_.reshape(-1, np.prod(shape[1:]))
    if x_.shape[1] > 5:
        x_ = x_[:, :6]
        x_val_ = x_val_[:, :6]
        x_test_ = x_test_[:, :6]
    x = pd.DataFrame(x_)
    x_val = pd.DataFrame(x_val_)
    x_test = pd.DataFrame(x_test_)
    y = pd.DataFrame(y_train[:, 0])
    y_val = pd.DataFrame(y_val[:, 0])
    y_test = pd.DataFrame(y_test[:, 0])
    lin_models = dict()
    total = 0
    for rule in act_train.columns:
        indices = act_train[rule].index[act_train[rule] >= params['thres_act']].tolist()
        if len(indices) != 0:
            X1 = x.loc[indices].values
            y1 = y.loc[indices].values

            lin_models[rule] = LinearRegression().fit(X1, y1.ravel())

    preds = pd.DataFrame(index=x_val.index, columns=sorted(lin_models.keys()))
    for rule in rules:
        indices = act_val[rule].index[act_val[rule] >= params['thres_act']].tolist()
        if len(indices) != 0 and rule in lin_models.keys():
            X1 = x_val.loc[indices].values
            preds.loc[indices, rule] = lin_models[rule].predict(X1).ravel()

    pred = preds.mean(axis=1)
    rated = y_val.values.ravel() if params['rated'] is None else 1
    err_val = (pred.values.ravel() - y_val.values.ravel()) /  rated

    preds = pd.DataFrame(index=x_test.index, columns=sorted(lin_models.keys()))
    for rule in rules:
        indices = act_test[rule].index[act_test[rule] >= params['thres_act']].tolist()
        if len(indices) != 0 and rule in lin_models.keys():
            X1 = x_test.loc[indices].values
            preds.loc[indices, rule] = lin_models[rule].predict(X1).ravel()

    pred = preds.mean(axis=1)
    rated = y_test.values.ravel() if params['rated'] is None else 1
    err_test = (pred.values.ravel() - y_test.values.ravel()) / rated
    mae_val = np.mean(np.abs(err_val)) + len_nan_val
    mae_test = np.mean(np.abs(err_test)) + len_nan_test
    sse_val = np.sum(np.square(err_val)) + len_nan_val
    sse_test = np.sum(np.square(err_test)) + len_nan_test
    return mae_val, mae_test, sse_val, sse_test


def find_free_cpus(path_group):
    free_cpus = 0
    warm = psutil.cpu_percent(percpu=True)

    while free_cpus < 2:
        sleep(5)
        load = psutil.cpu_percent(interval=None, percpu=True)
        n_cpus = len(load)
        available_load = n_cpus - int(n_cpus * np.mean(load) / 100) - 1
        print(f'find total {n_cpus} cpus,  mean load {np.mean(load)}, non_cpus {1}')
        free_cpus = n_cpus - 1
        print(
            f'Find load {int(n_cpus * np.mean(load) / 100)}, available {available_load} cpus,'
            f' {free_cpus} cpus free')
        if free_cpus > available_load:
            free_cpus = available_load
    print(f'Find {free_cpus} cpus free')
    return free_cpus


def send_predictions(message):
    contents = message
    # The mail addresses and password
    sender_address = 'gsdrts@yahoo.gr'
    sender_pass = 'pubmqkxfdtpqtwws'
    yag_smtp_connection = yagmail.SMTP(user=sender_address, password=sender_pass, host='smtp.mail.yahoo.com')
    subject = f'Error for check'
    yag_smtp_connection.send(to='joesider9@gmail.com', subject=subject, contents=contents)


def get_padding(images, max_w, max_h):
    imsize = images.shape[-3:-1]
    h_padding = (max_w - imsize[0]) / 2
    v_padding = (max_h - imsize[1]) / 2
    l_pad = h_padding if h_padding % 1 == 0 else h_padding + 0.5
    t_pad = v_padding if v_padding % 1 == 0 else v_padding + 0.5
    r_pad = h_padding if h_padding % 1 == 0 else h_padding - 0.5
    b_pad = v_padding if v_padding % 1 == 0 else v_padding - 0.5
    zero_pad = tuple([(0, 0) for _ in imsize])
    padding = zero_pad + ((int(l_pad), int(r_pad)), (int(t_pad), int(b_pad)))

    return padding


def pad_images(images, max_w=32, max_h=32, constant=0.0):
    panding = get_padding(images, max_w, max_h)
    return np.pad(images, panding, mode='constant', constant_values=constant)

def initialize_train_constants(params, epochs, len_performers=2):
    mae_old, sse_old = np.inf * np.ones(len_performers), np.inf * np.ones(len_performers)
    mae_max, sse_max = np.inf * np.ones(len_performers), np.inf * np.ones(len_performers)
    mae_min, sse_min = np.inf * np.ones(len_performers), np.inf * np.ones(len_performers)

    best_mae_val, best_mae_test = np.inf, np.inf
    train_flag, best_weights = True, None
    warm = params['warming_iterations']
    wait, best_iteration, best_tot_iteration, loops, n_iter = 0, 0, 0, 0, 0
    patience, exam_period = int(epochs / 3), int(epochs / 5)

    results_columns = ['Iteration', 'best_iteration', 'best_mae_val', 'best_mae_test', 'mae_val_out',
                       'mae_test_out', 'sse_val_out', 'sse_test_out']

    results = pd.DataFrame(columns=results_columns)
    return (mae_old, sse_old, mae_max, sse_max, mae_min, sse_min, results_columns, results, best_mae_val, best_mae_test,
            train_flag, best_weights, warm, wait, best_iteration, best_tot_iteration, loops,
            n_iter, patience, exam_period)

def initialize_fuzzy_train_constants(results_columns, epochs, len_performers=2):
    results_columns += ['sum_activations', 'min_activations', 'max_activations',
                        'mean_activations', 'mae_lin_val', 'mae_lin_test']
    if epochs <= 400:
        raise ValueError('epochs should be greater than 400 when it is fuzzy')
    mae_old_lin, sse_old_lin = np.inf * np.ones(len_performers), np.inf * np.ones(len_performers)
    mae_max_lin, sse_max_lin = np.inf * np.ones(len_performers), np.inf * np.ones(len_performers)
    mae_min_lin, sse_min_lin = np.inf * np.ones(len_performers), np.inf * np.ones(len_performers)
    results = pd.DataFrame(columns=results_columns)

    return mae_old_lin, sse_old_lin, mae_max_lin, sse_max_lin, mae_min_lin, sse_min_lin, results

def check_if_is_better(mae_old, mae_max, mae_min, sse_old, sse_max, sse_min,
                       mae_val, mae_test, sse_val, sse_test):
    mae = np.hstack([mae_val, mae_test])
    flag_mae, mae_old, mae_max, mae_min = distance(mae, mae_old, mae_max, mae_min)
    sse = np.hstack([sse_val, sse_test]) / 10
    flag_sse, sse_old, sse_max, sse_min = distance(sse, sse_old, sse_max, sse_min)
    flag_best = flag_mae or flag_sse
    if flag_best:
        mae_old, sse_old = mae, sse
    return mae_old, mae_max, mae_min, sse_old, sse_max, sse_min, flag_best

def store_results(results, results_columns, best_tot_iteration, n_iter, best_mae_val, best_mae_test, mae_val,
                  mae_test, sse_val, sse_test, fuzzy=False, sum_act=None, min_act=None, max_act=None, mean_act=None,
                                                       mae_val_lin=None, mae_test_lin=None):
    best_mae_val = mae_val if best_mae_val >= mae_val else best_mae_val
    best_mae_test = mae_test if best_mae_test >= mae_test else best_mae_test
    evaluation = np.array([n_iter, best_tot_iteration, best_mae_val, best_mae_test,
                           mae_val, mae_test, sse_val, sse_test])

    print_columns = ['best_mae_val', 'best_mae_test', 'mae_val_out', 'mae_test_out']
    if fuzzy:
        evaluation = np.concatenate([evaluation, np.array([sum_act,
                                                           min_act,
                                                           max_act, mean_act,
                                                           mae_val_lin, mae_test_lin])])
        print_columns += ['mae_lin_val', 'mae_lin_test']
    res = pd.DataFrame(evaluation.reshape(-1, 1).T, index=[n_iter], columns=results_columns)
    results = pd.concat([results, res])
    print(res[print_columns])
    return results, best_mae_val, best_mae_test


def check_if_extend_training(epochs, best_iteration, exam_period, patience, loops):
    train_flag = True
    if (epochs - best_iteration) <= exam_period:
        if loops > 3:
            train_flag = False
        else:
            epochs, exam_period = int(patience), int(patience / 5)
            patience = int(epochs / 1.5)
            best_iteration = 0
            loops += 1
    else:
        train_flag = False
    return epochs, best_iteration, exam_period, patience, loops, train_flag

def check_early_stop(wait, patience, epoch, epochs, sec_per_iter, sec_all):
    train_flag = True
    if wait > patience and epoch > (epochs / 2):
        train_flag = False
    if sec_per_iter > 3600:
        train_flag = False
    if sec_all > 60*60*12:
        train_flag = False
    return train_flag
