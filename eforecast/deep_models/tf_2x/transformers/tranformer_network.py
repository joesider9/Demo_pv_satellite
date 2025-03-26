import copy
import gc
import os
import random
import time

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
import yaml
from tqdm import tqdm

from eforecast.common_utils.train_utils import distance
from eforecast.common_utils.train_utils import remove_zeros_load_ts
from eforecast.deep_models.tf_2x.transformers.builders import build_graph
from eforecast.deep_models.tf_2x.optimizers import optimize
from eforecast.deep_models.tf_2x.transformers.trainer import gather_weights
from eforecast.deep_models.tf_2x.transformers.trainer import train_step
from eforecast.deep_models.tf_2x.transformers.trainer import validation_step
from eforecast.deep_models.tf_2x.transformers.trainer import feed_data_eval

pd.set_option('display.expand_frame_repr', False)


class TransformerNetwork:
    def __init__(self, static_data, path_weights, params=None, is_global=False, is_fuzzy=False, is_for_cluster=False,
                 probabilistic=False, refit=False):
        self.results = None
        self.best_sse_val = None
        self.best_sse_test = None
        self.best_min_act = None
        self.best_max_act = None
        self.best_mean_act = None
        self.best_sum_act = None
        self.best_mae_val = None
        self.best_mae_test = None
        self.best_weights = None
        self.n_batch = None
        self.n_out = None
        self.is_trained = False
        self.refit = refit
        self.probabilistic = probabilistic
        self.is_global = is_global
        self.is_fuzzy = is_fuzzy
        self.is_for_cluster = is_for_cluster
        self.static_data = static_data
        self.rated = static_data['rated']
        if params is not None:
            self.params = params
            self.method = self.params['method']
            self.name = self.params['name']
            self.model_layers = self.params['experiment']
            self.conv_dim = self.params.get('conv_dim')
            self.merge = self.params['merge']
            self.what_data = self.params['what_data']
            self.compress = self.params['compress']
            self.scale_nwp_method = self.params['scale_nwp_method']
            self.groups = self.params['groups']
            self.epochs = self.params['max_iterations']
            self.learning_rate = self.params['learning_rate']
            self.batch_size = self.params['batch_size']
        self.path_weights = path_weights

        try:
            if not self.refit:
                self.load()
        except:
            pass
        self.refit = refit
        if not hasattr(self, 'params'):
            raise RuntimeError('The network has no params. You should train the net providing params')

    def get_slice(self, x, mask, metadata, y=None):
        dates = metadata['dates']
        mask = mask.intersection(dates)
        indices = dates.get_indexer(mask)
        y_slice = y.iloc[indices].values if y is not None else None
        if isinstance(x, dict):  # These data is for lstm and mlp method
            X_slice = dict()
            for key in x.keys():
                if x[key].shape[0] > 0:
                    X_slice[key] = x[key][indices]
        else:
            raise ValueError('Wrong type of input X, Input should be dict with field observations, future calendar')
        return X_slice, y_slice, mask

    def fit(self, X, y, cv_masks, metadata, gpu_id=0):
        gpu_devices = tf.config.list_physical_devices('GPU')
        if len(gpu_devices) > 0:
            try:
                tf.config.set_visible_devices(gpu_devices[gpu_id], 'GPU')
                tf.config.experimental.set_memory_growth(gpu_devices[gpu_id], True)
            except RuntimeError as e:
                # Memory growth must be set before GPUs have been initialized
                print(e)
        print(tf.config.get_visible_devices(device_type='GPU'))
        y = remove_zeros_load_ts(self.static_data, y)
        if self.is_trained and not self.refit:
            return self.best_mae_test
        quantiles = self.params['quantiles'] if self.probabilistic else None

        self.params['n_out'] = y.shape[1]

        X_train, y_train, mask = self.get_slice(X, cv_masks[0], metadata, y=y)
        if self.name == 'global':
            cvs = 6
            cv_masks_val = np.array_split(cv_masks[1], cvs)
            X_val, y_val = [], []
            for cv_val in cv_masks_val:
                X_val_temp, y_val_temp, _ = self.get_slice(X, cv_val, metadata, y=y)
                X_val.append(X_val_temp)
                y_val.append(y_val_temp)
            cv_masks_test = np.array_split(cv_masks[2], cvs)
            X_test, y_test = [], []
            for cv_test in cv_masks_test:
                X_test_temp, y_test_temp, _ = self.get_slice(X, cv_test, metadata, y=y)
                X_test.append(X_test_temp)
                y_test.append(y_test_temp)
        else:
            cvs = 0
            X_val, y_val, _ = self.get_slice(X, cv_masks[1], metadata, y=y)
            X_test, y_test, _ = self.get_slice(X, cv_masks[2], metadata, y=y)

        with open(os.path.join(self.path_weights, 'parameters.txt'), 'w') as file:
            file.write(yaml.dump(self.params, default_flow_style=False, sort_keys=False))
        self.n_out = y_train.shape[1]
        N = cv_masks[0].intersection(mask).shape[0]
        self.batch_size = np.minimum(self.batch_size, int(N / 10))
        self.batch_size = np.maximum(self.batch_size, 32)
        self.n_batch = int(N / self.batch_size)

        print('Create graph....')
        tf.compat.v1.reset_default_graph()
        with tf.device('/device:GPU:' + str(gpu_id)):

            model_output = build_graph(X_train,
                                       self.model_layers,
                                       self.params,
                                       probabilistic=
                                       self.probabilistic,
                                       quantiles=quantiles)
            trainers, losses, MAEs, SSEs, learning_rate = optimize(is_global=False,
                                                                   rated=self.rated,
                                                                   learning_rate=1e-5,
                                                                   is_fuzzy=False,
                                                                   probabilistic=self.probabilistic,
                                                                   quantiles=quantiles,
                                                                   n_batch=self.n_batch,
                                                                   epochs=self.epochs)

        model_output.summary()
        len_performers = 2 if cvs == 0 else 2 * cvs
        mae_old, sse_old = np.inf * np.ones(len_performers), np.inf * np.ones(len_performers)
        mae_max, sse_max = np.inf * np.ones(len_performers), np.inf * np.ones(len_performers)
        mae_min, sse_min = np.inf * np.ones(len_performers), np.inf * np.ones(len_performers)

        best_mae_val, best_mae_test = np.inf, np.inf
        train_flag, best_weights = True, None
        warm = self.params['warming_iterations']
        wait, best_iteration, best_tot_iteration, loops, n_iter = 0, 0, 0, 0, 0
        epochs = self.epochs
        patience, exam_period = int(self.epochs / 6), int(self.epochs / 12)

        results_columns = ['Iteration', 'best_iteration', 'best_mae_val', 'best_mae_test', 'mae_val_out',
                           'mae_test_out', 'sse_val_out', 'sse_test_out']
        results = pd.DataFrame(columns=results_columns)

        print(f"Start training of {self.name} using {self.method} with {self.n_batch} batches and {self.epochs} epochs")

        while train_flag:
            for epoch in tqdm(range(epochs)):
                print(f'epoch {epoch}')
                train_step(model_output, losses, trainers, self.batch_size.astype(np.int64), X_train, y_train,
                           model_output.trainable_variables)

                warm = 0
                if cvs == 0:
                    sse_val = validation_step(model_output, SSEs, X_val, y_val).numpy()
                    mae_val = validation_step(model_output, MAEs, X_val, y_val).numpy()
                    sse_test = validation_step(model_output, SSEs, X_test, y_test).numpy()
                    mae_test = validation_step(model_output, MAEs, X_test, y_test).numpy()
                else:
                    sse_val = []
                    mae_val = []
                    for x_slice, y_slice in zip(X_val, y_val):
                        sse_val.append(validation_step(model_output, SSEs, x_slice, y_slice).numpy())
                        mae_val.append(validation_step(model_output, MAEs, x_slice, y_slice).numpy())
                    sse_val = np.array(sse_val)
                    mae_val = np.array(mae_val)
                    sse_test = []
                    mae_test = []
                    for x_slice, y_slice in zip(X_test, y_test):
                        sse_test.append(validation_step(model_output, SSEs, x_slice, y_slice).numpy())
                        mae_test.append(validation_step(model_output, MAEs, x_slice, y_slice).numpy())
                    sse_test = np.array(sse_test)
                    mae_test = np.array(mae_test)

                mae = np.hstack([mae_val, mae_test])
                flag_mae, mae_old, mae_max, mae_min = distance(mae, mae_old, mae_max, mae_min)
                sse = np.hstack([sse_val, sse_test])
                flag_sse, sse_old, sse_max, sse_min = distance(sse , sse_old, sse_max, sse_min)
                flag_best = flag_mae and flag_sse
                if flag_best:
                    # model_output.save()
                    best_weights = gather_weights(model_output)
                    best_tot_iteration = n_iter
                    best_iteration = epoch
                    wait = 0
                else:
                    wait += 1
                if cvs != 0:
                    mae_val = np.mean(mae_val)
                    mae_test = np.mean(mae_test)
                    sse_val = np.sum(sse_val)
                    sse_test = np.sum(sse_test)
                best_mae_val = mae_val if best_mae_val >= mae_val else best_mae_val
                best_mae_test = mae_test if best_mae_test >= mae_test else best_mae_test
                evaluation = np.array([n_iter, best_tot_iteration, best_mae_val, best_mae_test,
                                       mae_val, mae_test, sse_val, sse_test])

                if (best_mae_test > self.static_data['max_performance']) and epoch > 100:
                    self.best_mae_test = best_mae_test
                    self.save()
                    return
                print_columns = ['best_mae_val', 'best_mae_test', 'mae_val_out', 'mae_test_out']

                res = pd.DataFrame(evaluation.reshape(-1, 1).T, index=[n_iter], columns=results_columns)
                results = pd.concat([results, res])
                n_iter += 1
                print(res[print_columns])
                if wait > patience and epoch > (epochs / 2):
                    train_flag = False
                    break
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

        self.best_weights = best_weights
        self.best_mae_test = results['mae_test_out'].iloc[best_tot_iteration]
        self.best_mae_val = results['mae_val_out'].iloc[best_tot_iteration]
        self.best_sse_test = results['sse_test_out'].iloc[best_tot_iteration]
        self.best_sse_val = results['sse_val_out'].iloc[best_tot_iteration]
        self.results = results.iloc[best_tot_iteration]
        results.to_csv(os.path.join(self.path_weights, 'results.csv'))
        self.is_trained = True
        self.save()
        gc.collect()
        print(f"Total accuracy of validation: {self.best_mae_val} and of testing {self.best_mae_test}")

    def predict(self, X, metadata, cluster_dates=None, X_imp=None, activations=None, with_activations=False):
        self.load()
        quantiles = self.params['quantiles'] if self.probabilistic else None

        if not hasattr(self, 'best_weights'):
            raise NotImplementedError(f'The {self.method} network is not train. '
                                      f'The location path is {self.path_weights}')
        cluster_dates = metadata['dates'] if cluster_dates is None else cluster_dates.intersection(metadata['dates'])
        inp_x, _, cluster_dates = self.get_slice(X, cluster_dates, metadata)

        print('Create graph....')
        tf.compat.v1.reset_default_graph()
        with tf.device('/cpu:0'):
            model_output = build_graph(inp_x,
                                       self.model_layers,
                                       self.params,
                                       probabilistic=
                                       self.probabilistic,
                                       quantiles=quantiles)

        with tf.GradientTape() as tape:
            for variable in model_output.trainable_variables:
                variable.assign(self.best_weights[variable.name])
        x = feed_data_eval(inp_x)
        y_pred = model_output(x)
        y_pred = y_pred.numpy()
        if len(y_pred.shape) > 2:
            y_pred = np.squeeze(y_pred)
            y_pred = y_pred[:, 0].reshape(-1, 1)
        if self.static_data['horizon_type'] == 'multi-output':
            cols = [f'hour_ahead_{h}' for h in range(self.static_data['horizon'])]
        else:
            cols = [self.params['transformer_type']]
        if self.probabilistic:
            return y_pred
        else:
            y_pred = pd.DataFrame(y_pred, index=cluster_dates, columns=cols)
            if with_activations:
                activations = np.concatenate(activations[0], axis=1) if self.is_fuzzy else activations[0]
                activations = pd.DataFrame(activations, index=cluster_dates, columns=sorted(self.params['rules']))
                return y_pred, activations
            else:
                return y_pred

    def load(self):
        if os.path.exists(os.path.join(self.path_weights, 'net_weights.pickle')):
            try:
                tmp_dict = joblib.load(os.path.join(self.path_weights, 'net_weights.pickle'))
                self.__dict__.update(tmp_dict)
            except:
                raise ImportError(f'Cannot load weights for {self.method} model' + self.path_weights)
        else:
            raise ImportError(f'Cannot load weights for {self.method} model' + self.path_weights)

    def save(self):
        tmp_dict = {}
        for k in self.__dict__.keys():
            if k not in ['static_data', 'path_weights', 'refit']:
                tmp_dict[k] = self.__dict__[k]
        joblib.dump(tmp_dict, os.path.join(self.path_weights, 'net_weights.pickle'), compress=9)
