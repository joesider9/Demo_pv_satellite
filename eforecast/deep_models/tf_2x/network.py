import copy
import gc
import os
import random
import time

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
import traceback
import yaml
from openpyxl.styles.builtins import output
from tqdm import tqdm

from eforecast.datasets.data_feeder import DataFeeder
from eforecast.datasets.files_manager import FilesManager

from eforecast.common_utils.dataset_utils import load_data_deep_models
from eforecast.common_utils.train_utils import check_if_is_better
from eforecast.common_utils.train_utils import check_if_extend_training
from eforecast.common_utils.train_utils import store_results
from eforecast.common_utils.train_utils import check_early_stop
from eforecast.common_utils.dataset_utils import get_slice_for_nets
from eforecast.common_utils.train_utils import fix_convolutional_names
from eforecast.common_utils.train_utils import initialize_train_constants
from eforecast.common_utils.train_utils import initialize_fuzzy_train_constants
from eforecast.deep_models.tf_2x.global_builders import check_fuzzy_performance

from eforecast.deep_models.tf_2x.builders import build_graph
from eforecast.deep_models.tf_2x.global_builders import get_rbf
from eforecast.deep_models.tf_2x.global_builders import assign_rbf
from eforecast.deep_models.tf_2x.trainer import compute_tensors
from eforecast.deep_models.tf_2x.optimizers import optimize
from eforecast.deep_models.tf_2x.trainer import gather_weights
from eforecast.deep_models.tf_2x.trainer import train_schedule_fuzzy
from eforecast.deep_models.tf_2x.trainer import train_schedule_global
from eforecast.deep_models.tf_2x.trainer import train_step
from eforecast.deep_models.tf_2x.trainer import validation_step
from eforecast.deep_models.tf_2x.trainer import feed_data_eval

pd.set_option('display.expand_frame_repr', False)


class DeepNetwork:
    def __init__(self, static_data, path_weights, params=None, is_global=False, is_fuzzy=False, is_for_cluster=False,
                 probabilistic=False, train=False, online=False, refit=False):
        self.use_data = None
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
            self.data_types = self.params['data_types']
            self.conv_dim = self.params.get('conv_dim')
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
        self.is_online = online
        self.train = train
        self.data_feeder = DataFeeder(static_data, online=self.is_online, train=self.train)
        self.file_manager = FilesManager(static_data, is_online=self.is_online, train=self.train)
        if not hasattr(self, 'params'):
            raise RuntimeError('The network has no params. You should train the net providing params')

    def fit(self, cv_masks, gpu_id=0):
        gpu_devices = tf.config.list_physical_devices('GPU')
        if len(gpu_devices) > 0:
            try:
                tf.config.experimental.set_memory_growth(gpu_devices[gpu_id], True)
                tf.config.set_visible_devices(gpu_devices[gpu_id], 'GPU')
            except RuntimeError as e:
                # Memory growth must be set before GPUs have been initialized
                print(e)
        print(tf.config.get_visible_devices(device_type='GPU'))

        if self.is_trained and not self.refit:
            return self.best_mae_test
        quantiles = self.params['quantiles'] if self.probabilistic else None


        self.params['experiment'] = fix_convolutional_names(self.params['experiment_tag'], self.params['experiment'],
                                                            self.conv_dim)

        X, y, metadata, self.model_layers, self.params = load_data_deep_models(self.data_feeder, self.data_types,
                                                                               self.model_layers, self.params,
                                                                               self.train, self.is_fuzzy, self.refit)
        self.save()
        self.params['n_out'] = y.shape[1]
        X_train, y_train, mask = get_slice_for_nets(copy.deepcopy(X), cv_masks[0], metadata, y=y)
        X_val, y_val, mask_val = get_slice_for_nets(copy.deepcopy(X), cv_masks[1], metadata, y=y)
        X_test, y_test, mask_test = get_slice_for_nets(copy.deepcopy(X), cv_masks[2], metadata, y=y)

        with open(os.path.join(self.path_weights, 'parameters.txt'), 'w') as file:
            file.write(yaml.dump(self.params, default_flow_style=False, sort_keys=False))
        self.n_out = y_train.shape[1]
        N = cv_masks[0].intersection(mask).shape[0]
        self.n_batch = int(N / self.batch_size)
        print('Create graph....')
        tf.compat.v1.reset_default_graph()
        try:
            with tf.device('/device:GPU:' + str(gpu_id)):

                model_output, model_layers_built, act_all_tensor = build_graph(X_train,
                                                                               self.model_layers,
                                                                               self.params,
                                                                               is_fuzzy=self.is_fuzzy,
                                                                               is_global=self.is_global,
                                                                               is_for_cluster=
                                                                               self.is_for_cluster,
                                                                               probabilistic=
                                                                               self.probabilistic,
                                                                               quantiles=quantiles)
        except Exception as e:
            self.store_results_or_exit(error=e)
            return
        trainers, losses, MAEs, SSEs, learning_rate = optimize(is_global=self.is_global,
                                                                   rated=self.rated,
                                                                   learning_rate=self.learning_rate,
                                                                   is_fuzzy=self.is_fuzzy,
                                                                   probabilistic=self.probabilistic,
                                                                   quantiles=quantiles,
                                                                   n_batch=self.n_batch,
                                                                   epochs=self.epochs)

        model_output.summary()
        (mae_old, sse_old, mae_max, sse_max, mae_min, sse_min, results_columns, results, best_mae_val, best_mae_test,
         train_flag, best_weights, warm, wait, best_iteration, best_tot_iteration, loops,
         n_iter, patience, exam_period) = initialize_train_constants(self.params, self.epochs, len_performers=2)
        epochs = self.epochs
        if self.is_fuzzy:
            (mae_old_lin, sse_old_lin, mae_max_lin, sse_max_lin,
             mae_min_lin, sse_min_lin, results) = initialize_fuzzy_train_constants(results_columns,
                                                                                   epochs, len_performers=2)

        print(f"Start training of {self.name} using {self.method} with {self.n_batch} batches and {self.epochs} epochs")

        if self.is_global:
            init_clusters = get_rbf(model_output)
            best_clusters = get_rbf(model_output)

        while train_flag:
            for epoch in tqdm(range(epochs)):
                print(f'epoch {epoch}')
                start = time.time()

                try:
                    if not self.is_global and not self.is_fuzzy:
                        train_step(model_output, losses, trainers, self.batch_size.astype(np.int64), X_train, y_train,
                                   model_output.trainable_variables)
                    else:
                        if self.is_global and not self.is_fuzzy:
                            train_schedule_global(model_output, losses, trainers, self.batch_size, X_train,
                                                  y_train)
                        elif self.is_global and self.is_fuzzy:
                            train_schedule_fuzzy(model_output, losses, trainers, self.batch_size,
                                                 X_train, y_train, warm)
                except Exception as e:
                    self.store_results_or_exit(error=e)
                    return
                end = time.time()
                sec_per_iter = (end - start)
                warm = 0

                if self.is_fuzzy:
                    (net_model, sum_act,
                     min_act, max_act,
                     mean_act, warm, device,
                     mae_old_lin, mae_max_lin,
                     mae_min_lin, sse_old_lin,
                     sse_max_lin, sse_min_lin,
                     mae_val_lin, mae_test_lin) = check_fuzzy_performance(model_output, N, X_train, y_train, X_val, y_val,
                                                                          X_test, y_test, self.params, init_clusters,
                                                                          best_clusters, mae_old_lin,
                                                                          mae_max_lin, mae_min_lin, sse_old_lin,
                                                                          sse_max_lin, sse_min_lin,
                                                                          self.static_data['clustering'][
                                                                              'explode_clusters'])

                sse_val = validation_step(model_output, SSEs, X_val, y_val).numpy()
                mae_val = validation_step(model_output, MAEs, X_val, y_val).numpy()
                sse_test = validation_step(model_output, SSEs, X_test, y_test).numpy()
                mae_test = validation_step(model_output, MAEs, X_test, y_test).numpy()
                mae_old, mae_max, mae_min, sse_old, sse_max, sse_min, flag_best = check_if_is_better(mae_old, mae_max,
                                                                                                     mae_min, sse_old,
                                                                                                     sse_max, sse_min,
                                                                                                     mae_val, mae_test,
                                                                                                     sse_val, sse_test)
                if flag_best:
                    # model_output.save()
                    best_weights = gather_weights(model_output)
                    best_tot_iteration = n_iter
                    best_iteration = epoch
                    wait = 0
                else:
                    wait += 1

                if not self.is_fuzzy:
                    results, best_mae_val, best_mae_test = store_results(results, results_columns,
                                                                         best_tot_iteration,
                                                                         n_iter, best_mae_val, best_mae_test,
                                                                         mae_val,
                                                                         mae_test, sse_val, sse_test)
                else:
                    results, best_mae_val, best_mae_test = store_results(results, results_columns,
                                                                         best_tot_iteration,
                                                                         n_iter, best_mae_val, best_mae_test,
                                                                         mae_val,
                                                                         mae_test, sse_val, sse_test, fuzzy=True,
                                                                         sum_act=sum_act, min_act=min_act,
                                                                         max_act=max_act,
                                                                         mean_act=mean_act, mae_val_lin=mae_val_lin,
                                                                         mae_test_lin=mae_test_lin)
                n_iter += 1
                if (best_mae_test > self.static_data['max_performance']) and epoch > 100:
                    self.store_results_or_exit(best_iter=best_tot_iteration, results=results,
                                               best_weights=best_weights)
                    return
                train_flag = check_early_stop(wait, patience, epoch, epochs, sec_per_iter)
                if not train_flag:
                    break
            epochs, best_iteration, exam_period, patience, loops, train_flag = check_if_extend_training(epochs,
                                                                                                            best_iteration,
                                                                                                            exam_period,
                                                                                                            patience,
                                                                                                            loops)

        self.store_results_or_exit(best_iter=best_tot_iteration, results=results, best_weights=best_weights,
                                   store=True,
                                   fuzzy=self.is_fuzzy)
        gc.collect()

    def predict(self, cluster_dates=None, with_activations=False, columns=None):
        self.load()
        quantiles = self.params['quantiles'] if self.probabilistic else None

        if not hasattr(self, 'best_weights'):
            raise NotImplementedError(f'The {self.method} network is not train. '
                                      f'The location path is {self.path_weights}')
        X, y, metadata, self.model_layers, self.params = load_data_deep_models(self.data_feeder, self.data_types,
                                                                               self.model_layers, self.params,
                                                                               self.train, self.is_fuzzy, self.refit)
        inp_x, _, cluster_dates = get_slice_for_nets(X, cluster_dates, metadata)

        print('Create graph....')
        tf.compat.v1.reset_default_graph()
        with tf.device('/cpu:0'):
            model_output, model_layers_built, act_all_tensor = build_graph(inp_x,
                                                                           self.model_layers,
                                                                           self.params,
                                                                           is_fuzzy=self.is_fuzzy,
                                                                           is_global=self.is_global,
                                                                           is_for_cluster=
                                                                           self.is_for_cluster,
                                                                           probabilistic=
                                                                           self.probabilistic,
                                                                           quantiles=quantiles,
                                                                           train=False)

        with tf.GradientTape() as tape:
            for variable in model_output.trainable_variables:
                variable.assign(self.best_weights[variable.name])
        x = feed_data_eval(inp_x)
        y_pred = model_output(x)
        y_pred = y_pred.numpy()
        if len(y_pred.shape) > 2:
            y_pred = np.squeeze(y_pred)
            y_pred = y_pred[:, 0].reshape(-1, 1)
        if with_activations:
            activations = compute_tensors(model_output, 'activations', np.arange(y_pred.shape[0]), inp_x)
        if columns is None:
            if self.static_data['horizon_type'] == 'multi-output':
                cols = [f'hour_ahead_{h}' for h in range(self.static_data['horizon'])]
            else:
                cols = [self.method]
        else:
            cols = columns
        if self.probabilistic:
            return y_pred
        else:
            y_pred = pd.DataFrame(y_pred, index=cluster_dates, columns=cols)
            if with_activations:
                activations = pd.DataFrame(activations, index=cluster_dates, columns=sorted(self.params['rules']))
                return y_pred, activations
            else:
                return y_pred

    def store_results_or_exit(self, best_weights=None, results=None, best_iter=None, error=None, store=False,
                              fuzzy=False):
        if best_weights is None:
            best_weights = {}
        if best_iter is None and error is not None:
            tb = traceback.format_exception(error)
            print("".join(tb))
            with open(os.path.join(self.path_weights, 'error.txt'), mode='w') as fp:
                fp.write(" ".join(tb))
            self.best_mae_test, self.best_mae_val, self.best_sse_test, self.best_sse_val = np.inf, np.inf, np.inf, np.inf
            self.results, self.is_trained, self.best_weights = pd.DataFrame(), True, {}
            self.save()
        else:
            if len(best_weights) == 0:
                raise ValueError('Model weights cannot be empty')
            self.best_mae_test = results['mae_test_out'].iloc[best_iter]
            self.best_mae_val = results['mae_val_out'].iloc[best_iter]
            self.best_sse_test = results['sse_test_out'].iloc[best_iter]
            self.best_sse_val = results['sse_val_out'].iloc[best_iter]
            self.results, self.best_weights, self.is_trained = results.iloc[best_iter], best_weights, True
            if store:
                self.results.to_csv(os.path.join(self.path_weights, 'results.csv'))
                print(f"Total accuracy of validation: {self.best_mae_val} and of testing {self.best_mae_test}")
            if fuzzy:
                self.best_sum_act = results['sum_activations'].iloc[best_iter]
                self.best_min_act = results['min_activations'].iloc[best_iter]
                self.best_max_act = results['max_activations'].iloc[best_iter]
                self.best_mean_act = results['mean_activations'].iloc[best_iter]
                print(f'SUM OF ACTIVATIONS IS {self.best_sum_act}')
                print(f'MIN OF ACTIVATIONS IS {self.best_min_act}')
                print(f'MAX OF ACTIVATIONS IS {self.best_max_act}')
                print(f'MEAN OF ACTIVATIONS IS {self.best_mean_act}')

            self.save()

    def load(self):
        if os.path.exists(os.path.join(self.path_weights, 'net_weights.pickle')):
            try:
                tmp_dict = joblib.load(os.path.join(self.path_weights, 'net_weights.pickle'))
                self.__dict__.update(tmp_dict)
            except:
                raise ImportError('Cannot load weights for cnn model' + self.path_weights)
        else:
            raise ImportError('Cannot load weights for cnn model' + self.path_weights)

    def save(self):
        tmp_dict = {}
        for k in self.__dict__.keys():
            if k not in ['static_data', 'path_weights', 'refit', 'online', 'train',
                         'data_feeder', 'file_manager']:
                tmp_dict[k] = self.__dict__[k]
        joblib.dump(tmp_dict, os.path.join(self.path_weights, 'net_weights.pickle'), compress=9)