import copy
import gc
import os
import random
import time
import shutil

import joblib
import numpy as np
import pandas as pd
import torch

import traceback

from tqdm import tqdm

from eforecast.common_utils.train_utils import distance
from eforecast.common_utils.train_utils import fix_convolutional_names
from eforecast.deep_models.pytorch_2x.image.builders import build_graph
from eforecast.deep_models.pytorch_2x.image.optimizers import optimize
from eforecast.deep_models.pytorch_2x.image.trainer import train_step
from eforecast.deep_models.pytorch_2x.image.trainer import validation_step
from eforecast.deep_models.pytorch_2x.image.crossvivit_dataset import CrossvivitDataset
from eforecast.deep_models.pytorch_2x.image.crossvivit_dataset import CrossvivitDataloader

pd.set_option('display.expand_frame_repr', False)


class CrossViVitNetwork:
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


    def fit(self, gpu_id=0):
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

        self.params['experiment'] = fix_convolutional_names(self.params['experiment_tag'], self.params['experiment'],
                                                            self.conv_dim)

        self.params['group_layers'] = []
        x_sample = self.params['x_sample']
        n_batch_train = self.params['n_batch_train']
        n_batch_val = self.params['n_batch_val']
        n_batch_test = self.params['n_batch_test']
        path_dataset = self.params['path_dataset']
        self.params['n_out'] = x_sample['target'].shape[1]
        try:
            net_model = build_graph(x_sample, self.model_layers, self.params,
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

        len_performers = 2
        mae_old, sse_old = np.inf * np.ones(len_performers), np.inf * np.ones(len_performers)
        mae_max, sse_max = np.inf * np.ones(len_performers), np.inf * np.ones(len_performers)
        mae_min, sse_min = np.inf * np.ones(len_performers), np.inf * np.ones(len_performers)

        best_mae_val, best_mae_test = np.inf, np.inf
        train_flag, best_weights = True, None
        warm = self.params['warming_iterations']
        wait, best_iteration, best_tot_iteration, loops, n_iter = 0, 0, 0, 0, 0
        epochs = self.epochs
        patience, exam_period = 50, 15

        results_columns = ['Iteration', 'best_iteration', 'best_mae_val', 'best_mae_test', 'mae_val_out',
                           'mae_test_out', 'sse_val_out', 'sse_test_out']
        results = pd.DataFrame(columns=results_columns)

        print(f"Start training of {self.name} using {self.method} with {self.n_batch} batches and {self.epochs} epochs")

        while train_flag:
            for epoch in tqdm(range(epochs)):
                print(f'Start epoch {epoch}')
                try:
                    start = time.time()
                    train_step(net_model, loss, optimizers, device, path_dataset, n_batch_train)
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

                mae_val, sse_val = validation_step(net_model, [Accuracy, Sse], device,
                                                   path_dataset, n_batch_val, mode='val')
                mae_test, sse_test = validation_step(net_model, [Accuracy, Sse], device,
                                                     path_dataset, n_batch_test, mode='test')

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

    def predict(self, X, metadata, X_img, image_metadata, y, cluster_dates=None):
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

        X_train = CrossvivitDataset(self.static_data, copy.deepcopy(X), y, cluster_dates, image_metadata, cluster_dates,
                                    self.params, use_target=False)
        dataset = CrossvivitDataloader(X_train, 10, 10)

        x_sample = dataset.get_batch()
        T_pred = x_sample['future_nwp'].shape[1]
        cols = [f'Minutes_ahead_{15 * h}' for h in range(T_pred)]
        if np.isinf(self.best_mae_test):
            return pd.DataFrame(-999, index=cluster_dates, columns=cols)
        print('Create graph....')
        torch.cuda.empty_cache()
        gc.collect()
        batch_size = 512
        y_pred = []
        while batch_size >= 32:
            try:

                dataset = CrossvivitDataloader(X_train, batch_size, 18)
                with torch.no_grad():

                    y_pred = []
                    dates_eval = pd.DatetimeIndex([])
                    for idx in tqdm(range(dataset.n_batches)):
                        net_model = build_graph(x_sample, self.model_layers, self.params,
                                                probabilistic=self.probabilistic, quantiles=quantiles, device=device)
                        net_model.load_state_dict(self.best_weights)
                        net_model.to(device)
                        net_model.eval()
                        if idx == dataset.n_batches -1:
                            print(idx)
                        x = dataset.get_batch(device=device, randomly=False)
                        y_temp = net_model(x)[0]
                        y_temp = np.clip(y_temp.cpu().detach().numpy(), 0, None)
                        y_pred.append(y_temp)
                        dates_eval = dates_eval.append(x['dates'])
                        del y_temp
                        del net_model
                        torch.cuda.empty_cache()
                    break
            except:
                try:
                    del y_temp
                    del net_model
                except:
                    pass
                torch.cuda.empty_cache()
                gc.collect()
                batch_size = batch_size / 2
                continue

        if len(y_pred) == 0:
            return pd.DataFrame(-999, index=cluster_dates, columns=cols)
        else:
            y_pred1 = np.concatenate(y_pred, axis=0)
        if not self.probabilistic:
            y_pred1 = pd.DataFrame(y_pred1, index=dates_eval, columns=cols)
        try:
            del y_pred
            del y_temp
            del net_model
        except:
            pass
        torch.cuda.empty_cache()
        gc.collect()
        return y_pred1


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

