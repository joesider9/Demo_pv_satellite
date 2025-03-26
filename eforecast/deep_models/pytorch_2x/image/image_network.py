import copy
import gc
import os
import random
import time

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
import traceback
from tqdm import tqdm
from einops import rearrange

from eforecast.common_utils.train_utils import distance
from eforecast.common_utils.train_utils import pad_images
from eforecast.common_utils.train_utils import fix_convolutional_names
from eforecast.deep_models.pytorch_2x.image.builders import build_graph
from eforecast.deep_models.pytorch_2x.optimizers import optimize
from eforecast.deep_models.pytorch_2x.image.trainer import train_step
from eforecast.deep_models.pytorch_2x.image.trainer import validation_step
from eforecast.deep_models.pytorch_2x.image.trainer import feed_data_eval
from eforecast.deep_models.pytorch_2x.image.trainer import feed_dataset
from eforecast.common_utils.train_utils import send_predictions


pd.set_option('display.expand_frame_repr', False)


class ImageNetwork:
    def __init__(self, static_data, path_weights, params=None, probabilistic=False, refit=False):
        self.use_data = None
        self.results = None
        self.best_sse_val = None
        self.best_sse_test = None
        self.best_mae_val = None
        self.best_mae_test = None
        self.best_weights = None
        self.n_batch = None
        self.n_out = None
        self.is_trained = False
        self.refit = refit
        self.probabilistic = probabilistic
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

    def get_slice(self, x, x_img, mask, meta_data, img_meta_data, y=None):
        X = dict()
        x_img1 = []
        lat, long = self.params['site_indices']
        lags = self.params['temporal_lags']
        area_adjust = self.params['area_adjust']
        for image_type in self.params['image_type'].split(':'):
            x_img1.append(rearrange(x_img[image_type][:, 0:lags + 1, lat - area_adjust:lat + area_adjust,
                                                        long - area_adjust:long + area_adjust, :]
                                                        , 'b d1 w h d2 -> b w h (d1 d2)'))

        x_img = np.concatenate(x_img1, axis=-1)

        variables_cal = [var_data['name'] for var_data in self.static_data['variables']
                         if var_data['type'] == 'calendar']
        X['calendar'] = x[variables_cal]
        X['row_data'] = x.drop(columns=variables_cal)
        X['images'] = x_img
        X_temp = dict()
        if self.params['use_data'] == 1:
            self.use_data = {'calendar', 'row_data', 'image'}
        elif self.params['use_data'] == 2:
            self.use_data = {'calendar', 'image'}
        elif self.params['use_data'] == 3:
            self.use_data = {'images'}

        for key in X.keys():
            if key in self.use_data:
                X_temp[key] = X[key]
        shape = X_temp['images'].shape[-3:-1]

        x = copy.deepcopy(X_temp)
        group_layers = []
        dates = meta_data['dates']
        mask = mask.intersection(img_meta_data['dates'])
        mask = mask.intersection(dates)
        indices = dates.get_indexer(mask)
        indices_img = img_meta_data['dates'].get_indexer(mask)
        y_slice = y.iloc[indices].values if y is not None else None
        if len(self.groups) != 0:
            X_slice = dict()
            group_layers += self.merge.split('_')[1:]
            for group in self.groups:
                group_name = '_'.join(group) if isinstance(group, tuple) else group
                if isinstance(x[group_name], pd.DataFrame):
                    X_slice[group_name] = x[group_name].iloc[indices].values
                else:
                    data = x[group_name][indices_img]
                    X_slice[group_name] = data
        else:
            X_slice = dict()
            for key in x.keys():
                if isinstance(x[key], pd.DataFrame):
                    X_slice[key] = x[key].iloc[indices].values
                else:
                    data = x[key][indices_img]
                    X_slice[key] = data

        return X_slice, y_slice, group_layers, mask


    def fit(self, X, X_img, y, cv_masks, meta_data, image_metadata, gpu_id=0):
        if gpu_id != 'cpu':
            if torch.cuda.is_available():
                device = torch.device(f"cuda:{gpu_id}")
                print(f'Successfully find gpu cuda:{gpu_id}')
            else:
                print('Cannot find GPU device set cpu')
                device = torch.device("cpu")
        else:
            device = torch.device("cpu")
        if self.is_trained and not self.refit:
            return self.best_mae_test
        quantiles = self.params['quantiles'] if self.probabilistic else None

        self.params['n_out'] = y.shape[1]
        self.params['experiment'] = fix_convolutional_names(self.params['experiment_tag'], self.params['experiment'],
                                                            self.conv_dim)


        self.params['rules'] = None

        X_train, y_train, group_layers, mask = self.get_slice(copy.deepcopy(X), copy.deepcopy(X_img), cv_masks[0],
                                                              meta_data, image_metadata, y=y)
        X_val, y_val, _, mask_val = self.get_slice(copy.deepcopy(X), copy.deepcopy(X_img), cv_masks[1], meta_data,
                                                   image_metadata, y=y)
        X_test, y_test, _, mask_test = self.get_slice(copy.deepcopy(X), copy.deepcopy(X_img), cv_masks[2], meta_data,
                                                      image_metadata, y=y)

        if isinstance(X_train, dict):
            self.params['scopes'] = [scope for scope in X_train.keys() if 'act' not in scope]
        else:
            self.params['scopes'] = ['input']

        self.params['group_layers'] = group_layers
        with open(os.path.join(self.path_weights, 'parameters.txt'), 'w') as file:
            file.write(yaml.dump(self.params, default_flow_style=False, sort_keys=False))
        self.n_out = y_train.shape[1]
        N = cv_masks[0].intersection(mask).shape[0]
        N_val = cv_masks[1].intersection(mask_val).shape[0]
        N_test = cv_masks[2].intersection(mask_test).shape[0]
        self.batch_size = np.minimum(self.batch_size, int(N / 7.5))
        self.n_batch = int(self.batch_size)
        n_batch_val = int(self.batch_size)
        n_batch_test = int(self.batch_size)

        print('Create graph....')
        try:
            net_model = build_graph(X_train, self.model_layers, self.params,
                                    probabilistic=self.probabilistic, quantiles=quantiles, device=device)
        except Exception as e:
            tb = traceback.format_exception(e)
            print("".join(tb))
            with open(os.path.join(self.path_weights, 'error.txt'), mode='w') as fp:
                fp.write(" ".join(tb))
            self.best_weights = {}
            self.best_mae_test = np.inf
            self.best_mae_val = np.inf
            self.best_sse_test = np.inf
            self.best_sse_val = np.inf
            self.results = pd.DataFrame()
            self.is_trained = True
            self.save()
            return
        try:
            net_model.to(device)
        except Exception as e:
            tb = traceback.format_exception(e)
            print("".join(tb))
            with open(os.path.join(self.path_weights, 'error.txt'), mode='w') as fp:
                fp.write(" ".join(tb))
            self.best_weights = {}
            self.best_mae_test = np.inf
            self.best_mae_val = np.inf
            self.best_sse_test = np.inf
            self.best_sse_val = np.inf
            self.results = pd.DataFrame()
            self.is_trained = True
            self.save()
            return
        optimizers, schedulers, loss, Accuracy, Sse = optimize(net_model, device,
                                                               rated=self.rated,
                                                               learning_rate=self.learning_rate,
                                                               probabilistic=self.probabilistic,
                                                               quantiles=quantiles,
                                                               n_batch=self.n_batch)
        train_dataset = feed_dataset(X_train, y_train, batch_size=self.n_batch, shuffle=True)
        val_dataset = feed_dataset(X_val, y_val, batch_size=n_batch_val, shuffle=False)
        test_dataset = feed_dataset(X_test, y_test, batch_size=n_batch_test, shuffle=False)

        len_performers = 2
        mae_old, sse_old = np.inf * np.ones(len_performers), np.inf * np.ones(len_performers)
        mae_max, sse_max = np.inf * np.ones(len_performers), np.inf * np.ones(len_performers)
        mae_min, sse_min = np.inf * np.ones(len_performers), np.inf * np.ones(len_performers)

        best_mae_val, best_mae_test = np.inf, np.inf
        train_flag, best_weights = True, None
        warm = self.params['warming_iterations']
        wait, best_iteration, best_tot_iteration, loops, n_iter = 0, 0, 0, 0, 0
        epochs = self.epochs
        patience, exam_period = int(self.epochs / 3), int(self.epochs / 5)

        results_columns = ['Iteration', 'best_iteration', 'best_mae_val', 'best_mae_test', 'mae_val_out',
                           'mae_test_out', 'sse_val_out', 'sse_test_out']
        results = pd.DataFrame(columns=results_columns)

        print(f"Start training of {self.name} using {self.method} with {self.n_batch} batches and {self.epochs} epochs")

        while train_flag:
            for epoch in tqdm(range(epochs)):
                print(f'Start epoch {epoch}')
                try:
                    start = time.time()
                    train_step(net_model, loss, optimizers['bulk'], train_dataset, device)
                    end = time.time()
                    sec_per_iter = (end - start)
                except Exception as e:
                    tb = traceback.format_exception(e)
                    print("".join(tb))
                    with open(os.path.join(self.path_weights, 'error.txt'), mode='w') as fp:
                        fp.write(" ".join(tb))
                    self.best_weights = {}
                    self.best_mae_test = np.inf
                    self.best_mae_val = np.inf
                    self.best_sse_test = np.inf
                    self.best_sse_val = np.inf
                    self.results = pd.DataFrame()
                    self.is_trained = True
                    self.save()
                    return
                print(f'finish epoch {epoch}')
                sse_val = validation_step(net_model, Sse, val_dataset, device)
                mae_val = validation_step(net_model, Accuracy, val_dataset, device)
                sse_test = validation_step(net_model, Sse, test_dataset, device)
                mae_test = validation_step(net_model, Accuracy, test_dataset, device)

                for name_scheduler, scheduler in schedulers.items():
                    scheduler.step(mae_val + mae_test)
                sse_val = sse_val.cpu().detach().numpy()
                mae_val = mae_val.cpu().detach().numpy()
                sse_test = sse_test.cpu().detach().numpy()
                mae_test = mae_test.cpu().detach().numpy()
                mae = np.hstack([mae_val, mae_test])
                flag_mae, mae_old, mae_max, mae_min = distance(mae, mae_old, mae_max, mae_min)
                sse = np.hstack([sse_val, sse_test])
                flag_sse, sse_old, sse_max, sse_min = distance(sse / 10, sse_old, sse_max, sse_min)
                flag_best = flag_mae and flag_sse
                if flag_best:
                    # model_output.save()
                    best_weights = net_model.state_dict()
                    best_tot_iteration = n_iter
                    best_iteration = epoch
                    wait = 0
                else:
                    wait += 1
                best_mae_val = mae_val if best_mae_val >= mae_val else best_mae_val
                best_mae_test = mae_test if best_mae_test >= mae_test else best_mae_test
                evaluation = np.array([n_iter, best_tot_iteration, best_mae_val, best_mae_test,
                                       mae_val, mae_test, sse_val, sse_test])

                print_columns = ['best_mae_val', 'best_mae_test', 'mae_val_out', 'mae_test_out']

                res = pd.DataFrame(evaluation.reshape(-1, 1).T, index=[n_iter], columns=results_columns)
                results = pd.concat([results, res])
                n_iter += 1
                print(res[print_columns])
                if (best_mae_test > self.static_data['max_performance']) and epoch > 100:
                    self.best_weights = best_weights
                    self.best_mae_test = results['mae_test_out'].iloc[best_tot_iteration]
                    self.best_mae_val = results['mae_val_out'].iloc[best_tot_iteration]
                    self.best_sse_test = results['sse_test_out'].iloc[best_tot_iteration]
                    self.best_sse_val = results['sse_val_out'].iloc[best_tot_iteration]
                    self.results = results.iloc[best_tot_iteration]
                    self.is_trained = True
                    self.save()
                    return
                if wait > patience and epoch > (epochs / 2):
                    train_flag = False
                    break
                if sec_per_iter > 250:
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

    def predict(self, X, metadata, X_img, image_metadata, cluster_dates=None):
        if torch.cuda.is_available():
            device = torch.device(f"cuda:0")
            print(f'Successfully find gpu cuda:0')
        else:
            print('Cannot find GPU device set cpu')
            device = torch.device("cpu")
        self.load()
        quantiles = self.params['quantiles'] if self.probabilistic else None
        if self.what_data == 'row_dict_distributed':
            if not self.params['experiment']['input'][0][0] == 'lstm':
                raise ValueError('The first layer should be lstm when what data is row_dict_distributed')

        if not hasattr(self, 'best_weights'):
            raise NotImplementedError(f'The {self.method} network is not train. '
                                      f'The location path is {self.path_weights}')
        if self.static_data['horizon_type'] == 'multi-output':
            cols = [f'hour_ahead_{h}' for h in range(self.static_data['horizon'])]
        else:
            cols = [self.method]
        cluster_dates = metadata['dates'] if cluster_dates is None else cluster_dates.intersection(metadata['dates'])
        inp_x, _, _, cluster_dates = self.get_slice(X, X_img, cluster_dates, metadata, image_metadata)

        if np.isinf(self.best_mae_test):
            return pd.DataFrame(-999, index=cluster_dates, columns=cols)

        print('Create graph....')
        net_model = build_graph(inp_x, self.model_layers, self.params,
                                probabilistic=self.probabilistic, quantiles=quantiles, train=False, device=device)

        net_model.load_state_dict(self.best_weights)
        net_model.to(device)
        if len(cluster_dates) > self.batch_size:
            ind_range = np.arange(len(cluster_dates))
            inds = np.array_split(ind_range, int(len(cluster_dates)/self.batch_size) + 1)
        else:
            inds = np.arange(len(cluster_dates))
        with torch.no_grad():
            net_model.eval()
            y_pred = []
            for ind in inds:
                x = feed_data_eval(inp_x, ind=ind, device=device)
                y_temp = net_model(x)
                y_pred.append(y_temp.cpu().detach().numpy())

        y_pred = np.concatenate(y_pred, axis=0)
        if len(y_pred.shape) > 2:
            y_pred = np.squeeze(y_pred)
            y_pred = y_pred[:, 0].reshape(-1, 1)

        if self.probabilistic:
            return y_pred
        else:
            y_pred = pd.DataFrame(y_pred, index=cluster_dates, columns=cols)
            return y_pred

    def load(self):
        if os.path.exists(os.path.join(self.path_weights, 'net_weights.pickle')):
            try:
                tmp_dict = joblib.load(os.path.join(self.path_weights, 'net_weights.pickle'))
                self.__dict__.update(tmp_dict)
                self.best_weights = torch.load(os.path.join(self.path_weights, 'net_weights.pt'))
            except:
                raise ImportError('Cannot load weights for cnn model' + self.path_weights)
        else:
            raise ImportError('Cannot load weights for cnn model' + self.path_weights)

    def save(self):
        tmp_dict = {}
        for k in self.__dict__.keys():
            if k not in ['static_data', 'path_weights', 'refit', 'best_weights']:
                tmp_dict[k] = self.__dict__[k]
        joblib.dump(tmp_dict, os.path.join(self.path_weights, 'net_weights.pickle'))
        torch.save(self.best_weights, os.path.join(self.path_weights, 'net_weights.pt'))
