import copy
import gc
import os
import random
import time
import traceback

import joblib
import numpy as np
import pandas as pd
import torch

from tqdm import tqdm

from eforecast.common_utils.dataset_utils import get_slice_for_nets
from eforecast.datasets.data_feeder import DataFeeder
from eforecast.datasets.files_manager import FilesManager

from eforecast.common_utils.dataset_utils import load_data_deep_models
from eforecast.common_utils.train_utils import fix_convolutional_names
from eforecast.common_utils.train_utils import initialize_train_constants
from eforecast.common_utils.train_utils import initialize_fuzzy_train_constants
from eforecast.common_utils.train_utils import check_if_is_better
from eforecast.common_utils.train_utils import check_if_extend_training
from eforecast.common_utils.train_utils import store_results
from eforecast.common_utils.train_utils import check_early_stop
from eforecast.deep_models.pytorch_2x.builders import build_graph
from eforecast.deep_models.pytorch_2x.global_builders import get_rbf
from eforecast.deep_models.pytorch_2x.global_builders import check_fuzzy_performance
from eforecast.deep_models.pytorch_2x.trainer import compute_tensors
from eforecast.deep_models.pytorch_2x.optimizers import optimize
from eforecast.deep_models.pytorch_2x.trainer import train_schedule_fuzzy
from eforecast.deep_models.pytorch_2x.trainer import train_step
from eforecast.deep_models.pytorch_2x.trainer import validation_step
from eforecast.deep_models.pytorch_2x.trainer import feed_data_eval
from eforecast.deep_models.pytorch_2x.trainer import feed_dataset
from eforecast.deep_models.pytorch_2x.trainer import feed_image_dataset
from eforecast.deep_models.pytorch_2x.trainer import feed_image_dataset_real_time
from eforecast.datasets.image_data.image_dataset import ImageDataset
from eforecast.deep_models.pytorch_2x.image_dataset_real_time import ImageDatasetRealTime
from eforecast.datasets.image_data.image_dataset import ImageDataloader
from eforecast.common_utils.dataset_utils import get_data_dict_from_dates



pd.set_option('display.expand_frame_repr', False)


class DeepNetwork:
    def __init__(self, static_data, path_weights, params=None, is_global=False, is_fuzzy=False, is_for_cluster=False,
                 probabilistic=False, train=False, online=False, refit=False):
        self.use_data = None
        self.results = None
        self.best_sse_val = np.inf
        self.best_sse_test = np.inf
        self.best_min_act = np.inf
        self.best_max_act = np.inf
        self.best_mean_act = np.inf
        self.best_sum_act = np.inf
        self.best_mae_val = np.inf
        self.best_mae_test = np.inf
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
            self.params['rated'] = static_data['rated']
            self.method = self.params['method']
            self.name = self.params['name']
            self.model_layers = self.params['experiment']
            self.conv_dim = self.params.get('conv_dim')
            self.cluster = self.params.get('cluster', None)
            self.data_types = self.params['data_types']
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
            self.results, self.best_weights, self.is_trained = results, best_weights, True
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

    def fit(self, cv_masks, gpu_id=0):
        if gpu_id != 'cpu':
            if torch.cuda.is_available():
                device = torch.device(f"cuda:{gpu_id}")
                print(f'Successfully find gpu cuda:{gpu_id}')
            else:
                print('Cannot find GPU device set cpu')
                device = torch.device("cpu")
        else:
            device = torch.device("cpu")
        self.params['device'] = device
        if self.is_trained and not self.refit:
            return self.best_mae_test
        quantiles = self.params['quantiles'] if self.probabilistic else None

        self.params['experiment'] = fix_convolutional_names(self.static_data['horizon_type'], self.params['experiment'],
                                                            self.conv_dim)
        self.params['kernels'] = self.static_data['kernels']
        ## TODO Fix Image outputs in load_data_deep_models for Image2image case

        X, y, metadata, model_layers, self.params = load_data_deep_models(self.data_feeder, self.data_types,
                                                                           self.model_layers, self.params, self.cluster,
                                                                           self.train, self.is_fuzzy, self.refit)
        self.save()
        self.params['n_out'] = y.shape[1]
        X_train, y_train, mask = get_slice_for_nets(copy.deepcopy(X), metadata, dates=cv_masks[0], y=y)
        n_cvs = self.static_data['CVs']
        self.params['n_cvs'] = n_cvs

        X_val, y_val, mask_val = get_slice_for_nets(copy.deepcopy(X), metadata, dates=cv_masks[1], y=y)
        X_test, y_test, mask_test = get_slice_for_nets(copy.deepcopy(X), metadata, dates=cv_masks[2], y=y)

        self.n_out = y_train.shape[1]
        N = cv_masks[0].intersection(mask).shape[0]
        N_val = mask_val.shape[0]
        N_test = mask_test.shape[0]
        self.n_batch = int(self.batch_size)

        if 'images' not in self.model_layers:
            train_dataset = feed_dataset(X_train, y_train, device, batch_size=self.n_batch, shuffle=True)
            val_dataset = feed_dataset(X_val, y_val, device, batch_size=int(N_val / (2 * n_cvs)), shuffle=False)
            test_dataset = feed_dataset(X_test, y_test, device, batch_size=int(N_test / (2 * n_cvs)), shuffle=False)
        else:
            if self.params['create_image_dataset_real_time']:
                train_dataset = feed_image_dataset_real_time(self.static_data, X_train, y_train, mask, self.params,
                                                             device,
                                                             batch_size=self.batch_size, shuffle=True,
                                                             train=True, use_target=False)
                val_dataset = feed_image_dataset_real_time(self.static_data, X_val, y_val, mask_val, self.params,
                                                           device,
                                                           batch_size=self.batch_size, shuffle=False,
                                                           train=True, use_target=False)
                test_dataset = feed_image_dataset_real_time(self.static_data, X_test, y_test, mask_test, self.params,
                                                            device,
                                                            batch_size=self.batch_size, shuffle=False,
                                                            train=True, use_target=False)
            else:
                path_dataset = os.path.join(self.static_data['path_image'], 'SAT_DATA',
                                            self.params['sat_image_type'].replace(':', '_'),
                                            f'gpu_id_{gpu_id}')
                train_dataset = feed_image_dataset(X_train, y_train, mask, 'train', path_dataset, device)
                val_dataset = feed_image_dataset(X_val, y_val, mask_val, 'val', path_dataset, device)
                test_dataset = feed_image_dataset(X_test, y_test, mask_test, 'test', path_dataset, device)

        print('Create graph....')
        try:
            from eforecast.common_utils.devops_utils import get_cuda_available_memory
            mem_init = get_cuda_available_memory()
            mem_alloc_init, mem_res_init = torch.cuda.memory_allocated() / 2 ** 30, torch.cuda.memory_reserved() / 2 ** 30
            x_sample = next(iter(train_dataset))[0]
            if 'images' in self.model_layers and not self.params['create_image_dataset_real_time']:
                x_sample['images'] = self.params['x_sample']['images'].numpy().squeeze()
            net_model = build_graph(x_sample, model_layers, self.params, is_fuzzy=self.is_fuzzy,
                                    probabilistic=self.probabilistic, quantiles=quantiles, device=device)
            from accelerate.utils import calculate_maximum_sizes

            total_size, largest_layer = calculate_maximum_sizes(net_model)
            num_parameters = sum(p.numel() for p in net_model.parameters() if p.requires_grad)
            dtype_total_size = total_size

            dtype_training_size = dtype_total_size * 4 / (1024 ** 3)
            dtype_inference = dtype_total_size * 1.2 / (1024 ** 3)
            dtype_total_size = dtype_total_size / (1024 ** 3)
            gpu_data = {"Total Size (GB)": dtype_total_size,
                        "Inference (GB)": dtype_inference,
                        "Training using Adam (GB)": dtype_training_size,
                        "Parameters (Billion)": num_parameters / 1e9}
            if 'tl_path_weights' in self.params:
                self.best_weights = torch.load(os.path.join(self.params['tl_path_weights'], 'net_weights.pt'),
                                               weights_only=False)
                net_model.load_state_dict(self.best_weights)
                self.transfer_learning_trained = True
            else:
                self.transfer_learning_trained = False
            net_model.to(device)
            mem_net_device = get_cuda_available_memory()
            mem_alloc_device, mem_res_device = torch.cuda.memory_allocated() / 2 ** 30, torch.cuda.memory_reserved() / 2 ** 30
            gpu_data['mem_init'] = [mem_init, mem_alloc_init, mem_res_init]
            gpu_data['mem_net_device'] = [mem_net_device, mem_alloc_device, mem_res_device]
        except Exception as e:
            self.store_results_or_exit(error=e)
            return
        optimizers, schedulers, loss, Accuracy, Sse = optimize(net_model, device,
                                                               optimizer=self.params.get('optimizer', 'adam'),
                                                               scheduler=self.params.get('scheduler', 'CosineAnnealing'),
                                                               rated=self.rated,
                                                               learning_rate=self.learning_rate,
                                                               is_fuzzy=self.is_fuzzy,
                                                               probabilistic=self.probabilistic,
                                                               quantiles=quantiles)





        (mae_old, sse_old, mae_max, sse_max, mae_min, sse_min, results_columns, results, best_mae_val, best_mae_test,
         train_flag, best_weights, warm, wait, best_iteration, best_tot_iteration, loops,
         n_iter, patience, exam_period) = initialize_train_constants(self.params, self.epochs, len_performers=2 * n_cvs)
        epochs = self.epochs
        if self.is_fuzzy:
            (mae_old_lin, sse_old_lin, mae_max_lin, sse_max_lin,
             mae_min_lin, sse_min_lin, results) = initialize_fuzzy_train_constants(results_columns,
                                                                                   epochs, len_performers=2)

        print(f"Start training of {self.name} using {self.method} with {self.n_batch} batches and {self.epochs} epochs")

        if self.is_global or self.is_fuzzy:
            init_clusters = get_rbf(net_model)
            best_clusters = get_rbf(net_model)

        gpu_data['mem_train'] = []
        gpu_data['mem_val'] = []
        start_time = time.time()
        while train_flag:
            for epoch in tqdm(range(epochs)) :
                print(f'Training model: epoch {epoch}')
                start = time.time()
                if epoch == 0 and self.transfer_learning_trained:
                    # Skip the first epoch if it is transfer learning, pretrained weights have been applied
                    pass
                else:
                    try:
                        if not self.is_fuzzy:
                            train_step(net_model, loss, optimizers['bulk'], train_dataset, device)
                        else:
                            train_schedule_fuzzy(net_model, loss, optimizers, train_dataset, device, warm)
                    except Exception as e:
                        self.store_results_or_exit(error=e)
                        return
                end = time.time()
                sec_per_iter = (end - start)
                warm = 0
                avail_mem = get_cuda_available_memory()
                mem_alloc_train, mem_res_train = torch.cuda.memory_allocated() / 2 ** 30, torch.cuda.memory_reserved() / 2 ** 30
                gpu_data['mem_train'].append([avail_mem, mem_alloc_train, mem_res_train])
                if self.is_fuzzy:
                    (net_model, best_clusters, sum_act,
                     min_act, max_act, mean_act,
                     warm, mae_old_lin, mae_max_lin,
                     mae_min_lin, sse_old_lin,
                     sse_max_lin, sse_min_lin,
                     mae_val_lin, mae_test_lin) = check_fuzzy_performance(net_model, N, X_train, y_train, X_val, y_val,
                                                                         X_test, y_test, self.params, init_clusters,
                                                                         best_clusters, device, mae_old_lin,
                                                                         mae_max_lin, mae_min_lin, sse_old_lin,
                                                                         sse_max_lin, sse_min_lin,
                                                                         self.static_data['clustering']['explode_clusters'])

                mae_val, sse_val = validation_step(net_model, Accuracy, Sse, val_dataset, device, n_cvs=n_cvs)
                mae_test, sse_test = validation_step(net_model, Accuracy, Sse, test_dataset, device, n_cvs=n_cvs)
                avail_mem = get_cuda_available_memory()
                mem_alloc_train, mem_res_train = torch.cuda.memory_allocated() / 2 ** 30, torch.cuda.memory_reserved() / 2 ** 30
                gpu_data['mem_val'].append([avail_mem, mem_alloc_train, mem_res_train])
                for name_scheduler, scheduler in schedulers.items():
                    if name_scheduler == 'fuzzy':
                        scheduler.step()
                    else:
                        if np.isnan(np.mean(mae_val)) or np.isnan(np.mean(mae_test)):
                            raise ValueError('NaN in mae_val or mae_test')
                        scheduler.step(np.mean(mae_val) + np.mean(mae_test))
                mae_old, mae_max, mae_min, sse_old, sse_max, sse_min, flag_best = check_if_is_better(mae_old, mae_max,
                                                                                                     mae_min, sse_old,
                                                                                                     sse_max, sse_min,
                                                                                                     mae_val, mae_test,
                                                                                                     sse_val, sse_test)
                if flag_best:
                    # model_output.save()
                    best_weights = net_model.state_dict()
                    best_tot_iteration = n_iter
                    best_iteration = epoch
                    wait = 0
                else:
                    wait += 1
                mae_val, mae_test, sse_val, sse_test = (np.mean(mae_val), np.mean(mae_test), np.mean(sse_val),
                                                        np.mean(sse_test))
                if not self.is_fuzzy:
                    results, best_mae_val, best_mae_test = store_results(results, results_columns, best_tot_iteration,
                                                                         n_iter, best_mae_val, best_mae_test, mae_val,
                                                                         mae_test, sse_val, sse_test)
                else:
                    results, best_mae_val, best_mae_test = store_results(results, results_columns, best_tot_iteration,
                                                                         n_iter, best_mae_val, best_mae_test, mae_val,
                                                                         mae_test, sse_val, sse_test, fuzzy=True,
                                                                         sum_act=sum_act, min_act=min_act, max_act=max_act,
                                                                         mean_act=mean_act, mae_val_lin=mae_val_lin,
                                                                         mae_test_lin=mae_test_lin)
                n_iter += 1
                if (best_mae_test > self.static_data['max_performance']) and epoch > 100:
                    self.store_results_or_exit(best_iter=best_tot_iteration, results=results,
                                               best_weights=best_weights)
                    return
                end_time = time.time()
                train_flag = check_early_stop(wait, patience, epoch, epochs, sec_per_iter, end_time - start_time)
                if not train_flag:
                    break
            epochs, best_iteration, exam_period, patience, loops, train_flag = check_if_extend_training(epochs,
                                                                                                        best_iteration,
                                                                                            exam_period, patience,
                                                                                            loops)

        self.store_results_or_exit(best_iter=best_tot_iteration, results=results, best_weights=best_weights, store=True,
                                   fuzzy=self.is_fuzzy)
        joblib.dump(gpu_data, os.path.join(self.path_weights, 'gpu_data.pickle'))
        gc.collect()

    def predict_image(self, cluster_dates=None):
        if torch.cuda.is_available():
            device = torch.device(f"cuda:0")
            print(f'Successfully find gpu cuda:0')
        else:
            print('Cannot find GPU device set cpu')
            device = torch.device("cpu")
        quantiles = self.params['quantiles'] if self.probabilistic else None

        if not hasattr(self, 'best_weights'):
            raise NotImplementedError(f'The {self.method} network is not train. '
                                      f'The location path is {self.path_weights}')

        cluster = self.params.get('cluster', None)

        X, metadata, model_layers, self.params = load_data_deep_models(self.data_feeder, self.data_types,
                                                                               self.model_layers, self.params, cluster,
                                                                               False, self.is_fuzzy, False)
        area_adjust = int(self.params['area_adjust'])
        path_dataset = os.path.join(self.static_data['path_image'], 'SAT_DATA',
                                    self.params['sat_image_type'].replace(':', '_'),
                                    f'gpu_id_0')
        if self.train:
            cv_masks = joblib.load(os.path.join(self.cluster['cluster_path'], 'cv_mask.pickle'))
            X_val, _, mask_val = get_slice_for_nets(copy.deepcopy(X), metadata, dates=cv_masks[1])
            X_test, _, mask_test = get_slice_for_nets(copy.deepcopy(X), metadata, dates=cv_masks[2])
            if self.params['create_image_dataset_real_time']:
                val_dataset = feed_image_dataset_real_time(self.static_data, X_val, None, mask_val, self.params,
                                                             device,
                                                             batch_size=self.batch_size, shuffle=False,
                                                             train=False, use_target=False)
                test_dataset = feed_image_dataset_real_time(self.static_data, X_test, None, mask_test, self.params,
                                                             device,
                                                             batch_size=self.batch_size, shuffle=False,
                                                             train=False, use_target=False)
            else:
                val_dataset = feed_image_dataset(X_val, None, mask_val, 'val', path_dataset, device,
                                                   train=False)
                test_dataset = feed_image_dataset(X_test, None, mask_test, 'test', path_dataset, device,
                                                   train=False)
            cluster_dates = mask_val.union(mask_test)

            dataset = [val_dataset, test_dataset]
        elif not self.train and not self.is_online:
            X_eval, _, mask = get_slice_for_nets(copy.deepcopy(X), metadata, dates=cluster_dates)
            cluster_dates = mask
            if self.params['create_image_dataset_real_time']:
                eval_dataset = feed_image_dataset_real_time(self.static_data, X_eval, None, mask,
                                                            self.params, device,
                                                        batch_size=self.batch_size, shuffle=False,
                                                        train=False, use_target=False)
            else:
                image_dataset = ImageDataset(self.static_data, mask, self.params, use_target=False)
                eval_image_dataset = ImageDataloader(image_dataset, int(self.batch_size), 1)
                path_dataset = os.path.join(self.static_data['path_data'], f"area_adjust_{area_adjust}", f'gpu_id_0')

                if not os.path.exists(path_dataset):
                    os.makedirs(path_dataset)
                print('begin to create batches....')
                n_batch_train = 0
                real_length = 0
                for idx in tqdm(range(eval_image_dataset.n_batches)):
                    if not os.path.exists(f"{path_dataset}/eval_tensor{idx}.pt"):
                        x_batch = eval_image_dataset.get_batch()
                        real_length += x_batch["dates"].shape[0]
                        torch.save(x_batch, f"{path_dataset}/eval_tensor{idx}.pt")
                        if not eval_image_dataset.valid:
                            break
                        print(f'Train batch #{idx}: written')
                    n_batch_train += 1

                eval_dataset = feed_image_dataset(X_eval, None, mask, 'eval', path_dataset, device,
                                                  train=False)
            dataset = [eval_dataset]
        else:
            inp_x, _, cluster_dates = get_slice_for_nets(X, metadata, dates=cluster_dates)
            if self.params['create_image_dataset_real_time']:
                dataset = feed_image_dataset_real_time(self.static_data, inp_x, None, cluster_dates,
                                                            self.params, device,
                                                            batch_size=1, shuffle=False,
                                                            train=False, use_target=False, is_online=self.is_online)
            else:
                image_dataset = ImageDataset(self.static_data, cluster_dates, self.params, use_target=False)
                eval_image_dataset = ImageDataloader(image_dataset, int(self.batch_size), 1)
                x_batch = eval_image_dataset.get_batch()
                dates_sat = pd.DatetimeIndex([d[0] for d in x_batch['dates']])
                dates = dates_sat.intersection(cluster_dates)
                indices = dates_sat.get_indexer(dates)
                inp_x = get_data_dict_from_dates(inp_x, dates, dates_dict=cluster_dates)
                inp_x = feed_data_eval(inp_x)
                inp_x['images'] = x_batch['images'][indices].squeeze(0)
                dataset = inp_x

        cols = [f'{self.method}_{col}' for col in self.static_data['target_variable']['columns']]
        print('Create graph....')
        if np.isinf(self.best_mae_test):
            return pd.DataFrame(-999, index=cluster_dates, columns=cols)

        torch.cuda.empty_cache()
        gc.collect()
        with torch.inference_mode():
            y_pred = []
            dates_eval = pd.DatetimeIndex([])
            if not isinstance(dataset, list):
                if self.params['create_image_dataset_real_time']:
                    inp_x, dates = next(iter(dataset))
                    dates = pd.DatetimeIndex([pd.to_datetime(timestamp_tensor.numpy(), unit='s').round('15min')
                                                  for timestamp_tensor in dates])
                else:
                    inp_x, dates = dataset, cluster_dates
                net_model = build_graph(inp_x, model_layers, self.params,
                                        probabilistic=self.probabilistic, quantiles=quantiles, device=device)
                net_model.load_state_dict(self.best_weights)
                net_model.to(device)
                net_model.eval()
                y_temp = net_model(dataset)
                y_pred = np.clip(y_temp.cpu().detach().numpy(), 0, None)
                dates_eval = dates
            else:
                for dset in dataset:
                    for inp_x, dates in tqdm(dset):
                        if inp_x is None:
                            continue
                        if self.params['create_image_dataset_real_time']:
                            dates = pd.DatetimeIndex([pd.to_datetime(timestamp_tensor.numpy(), unit='s').
                                                     round(self.static_data['ts_resolution'])
                                                      for timestamp_tensor in dates])
                        net_model = build_graph(inp_x, model_layers, self.params,
                                                probabilistic=self.probabilistic, quantiles=quantiles, device=device)
                        net_model.load_state_dict(self.best_weights)
                        net_model.to(device)
                        net_model.eval()
                        y_temp = net_model(inp_x)
                        y_temp = np.clip(y_temp.cpu().detach().numpy(), 0, None)
                        y_pred.append(y_temp)
                        dates_eval = dates_eval.append(dates)
                        del y_temp
                        del net_model
                        torch.cuda.empty_cache()
                y_pred = np.concatenate(y_pred, axis=0)


        if self.probabilistic:
            return y_pred
        else:
            y_pred = pd.DataFrame(y_pred, index=dates_eval, columns=cols).sort_index()
            try:
                del net_model
            except:
                pass
            torch.cuda.empty_cache()
            gc.collect()
            return y_pred



    def predict(self, cluster_dates=None, with_activations=False):
        if 'images' in self.model_layers:
            return self.predict_image(cluster_dates)
        quantiles = self.params['quantiles'] if self.probabilistic else None

        if not hasattr(self, 'best_weights'):
            raise NotImplementedError(f'The {self.method} network is not train. '
                                      f'The location path is {self.path_weights}')

        cluster = self.params.get('cluster', None)

        X, metadata, model_layers, self.params = load_data_deep_models(self.data_feeder, self.data_types,
                                                                               self.model_layers, self.params, cluster,
                                                                               False, self.is_fuzzy, False)
        inp_x, _, cluster_dates = get_slice_for_nets(X, metadata, dates=cluster_dates)
        cols = [f'{self.method}_{col}' for col in self.static_data['target_variable']['columns']]
        print('Create graph....')
        if np.isinf(self.best_mae_test):
            return pd.DataFrame(-999, index=cluster_dates, columns=cols)

        net_model = build_graph(inp_x, model_layers, self.params, is_fuzzy=self.is_fuzzy,
                                probabilistic=self.probabilistic, quantiles=quantiles, train=False)

        net_model.load_state_dict(self.best_weights)
        net_model.eval()
        x = feed_data_eval(inp_x)
        y_pred = net_model(x)
        y_pred = y_pred.detach().numpy()

        if with_activations:
            activations = compute_tensors(net_model, x).detach().numpy()

        if self.probabilistic:
            return y_pred
        else:
            y_pred = pd.DataFrame(y_pred, index=cluster_dates, columns=cols)
            if with_activations:
                activations = pd.DataFrame(activations, index=cluster_dates, columns=sorted(self.params['rules']))
                return y_pred, activations
            else:
                return y_pred

    def load(self):
        if os.path.exists(os.path.join(self.path_weights, 'net_weights.pickle')):
            try:
                tmp_dict = joblib.load(os.path.join(self.path_weights, 'net_weights.pickle'))
                self.__dict__.update(tmp_dict)
                self.best_weights = torch.load(os.path.join(self.path_weights, 'net_weights.pt'), weights_only=False,
                                               map_location=torch.device('cpu'))
            except:
                raise ImportError('Cannot load weights for deep model' + self.path_weights)
        else:
            raise ImportError('Cannot load weights for deep model' + self.path_weights)

    def save(self):
        tmp_dict = {}
        for k in self.__dict__.keys():
            if k not in ['static_data', 'path_weights', 'refit', 'best_weights']:
                tmp_dict[k] = self.__dict__[k]
        joblib.dump(tmp_dict, os.path.join(self.path_weights, 'net_weights.pickle'))
        torch.save(self.best_weights, os.path.join(self.path_weights, 'net_weights.pt'))
