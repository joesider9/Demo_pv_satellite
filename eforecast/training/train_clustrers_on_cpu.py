import os
import gc
import glob
import time
import shutil
import joblib
import optuna
import  traceback

import numpy as np
import pandas as pd

from optuna.samplers import TPESampler

from eforecast.common_utils.dataset_utils import sync_datasets
from eforecast.common_utils.dataset_utils import sync_dict_df
from eforecast.common_utils.train_utils import find_free_cpus
from eforecast.common_utils.train_utils import send_predictions


from eforecast.datasets.data_feeder import DataFeeder
from eforecast.shallow_models.shallow_model import ShallowModel

from sklearn.linear_model import LassoCV, Lasso
from sklearn.linear_model import MultiTaskLassoCV, MultiTaskLasso
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor


class Objective(object):
    def __init__(self, static_data, cluster_name, cluster_dir, method, n_jobs):
        self.static_data = static_data
        self.method = method
        self.cluster_dir = cluster_dir
        self.cluster_name = cluster_name
        self.n_jobs = n_jobs

        self.nwp_data_merge = self.static_data['nwp_data_merge']
        self.nwp_data_compress = self.static_data['compress_data']
        self.scale_nwp_methods = self.static_data['scale_nwp_method']
        self.scale_row_methods = self.static_data['scale_row_method']
        self.feature_selection_methods = self.static_data['feature_selection_methods']
        self.data_feeder = DataFeeder(self.static_data, online=False, train=True)

    def __call__(self, trial):
        param_init = find_init_params(self.static_data, self.method)
        if len(self.nwp_data_merge) > 1:
            merge = trial.suggest_categorical('merge', self.nwp_data_merge)
        else:
            merge = list(self.nwp_data_merge)[0]
        if len(self.nwp_data_compress) > 1:
            compress = trial.suggest_categorical('compress', self.nwp_data_compress)
        else:
            compress = list(self.nwp_data_compress)[0]
        if len(self.feature_selection_methods) > 1:
            feature_selection_method = trial.suggest_categorical('feature_selection_method',
                                                                 self.feature_selection_methods)
        else:
            feature_selection_method = self.feature_selection_methods[0]
        what_data = 'row_all'

        if len(self.scale_nwp_methods) > 1:
            scale_nwp_method = trial.suggest_categorical('scale_nwp_method', self.scale_nwp_methods)
        else:
            scale_nwp_method = list(self.scale_nwp_methods)[0]
        if len(self.scale_row_methods) > 1:
            scale_row_method = trial.suggest_categorical('scale_row_method', self.scale_row_methods)
        else:
            scale_row_method = list(self.scale_row_methods)[0]

        cv_masks = joblib.load(os.path.join(self.cluster_dir, 'cv_mask.pickle'))
        cv_masks = [cv_masks[i] for i in [0, 2, 1]]
        experiment_params = {'method': self.method,
                             'cluster_name': self.cluster_name,
                             'cluster_dir': self.cluster_dir,
                             'merge': merge,
                             'compress': compress,
                             'what_data': what_data,
                             'feature_selection_method': feature_selection_method,
                             'scale_nwp_method': scale_nwp_method,
                             'scale_row_method': scale_row_method}
        for param, value in self.static_data[self.method].items():
            if param == 'depth':
                continue
            if isinstance(value, set):
                if param in param_init.keys():
                    value.add(param_init[param])
                if len(value) > 1:
                    v = trial.suggest_categorical(param, list(value))
                else:
                    v = list(value)[0]
            elif isinstance(value, list):
                if len(value) > 1:
                    if param in param_init.keys():
                        if param_init[param] is not None:
                            if param_init[param] < value[0]:
                                value[0] = param_init[param]
                            if param_init[param] > value[1]:
                                value[1] = param_init[param]
                    if isinstance(value[0], int):
                        v = trial.suggest_int(param, value[0], value[-1])
                    else:
                        v = trial.suggest_float(param, value[0], value[-1])
                else:
                    v = value[0]
            else:
                v = value
            experiment_params[param] = v
        if 'depth' in self.static_data[self.method].keys():
            param = 'depth'
            value = self.static_data[self.method][param]
            if 'boosting_type' in experiment_params.keys():
                if experiment_params["boosting_type"] == "Ordered" and self.static_data['horizon_type'] == 'multi-output':
                    experiment_params["depth"] = trial.suggest_int(param, value[0], 6)
                else:
                    experiment_params["depth"] = trial.suggest_int(param, value[0], value[-1])
            else:
                experiment_params["depth"] = trial.suggest_int(param, value[0], value[-1])
        if 'bootstrap_type' in experiment_params.keys():
            if experiment_params["bootstrap_type"] == "Bayesian":
                experiment_params["bagging_temperature"] = 6
            elif experiment_params["bootstrap_type"] == "Bernoulli":
                experiment_params["subsample"] = 0.5

        path_weights = os.path.join(self.cluster_dir, self.method, f'test_{trial.number}')
        if not os.path.exists(path_weights):
            os.makedirs(path_weights)
        else:
            shutil.rmtree(path_weights)
            os.makedirs(path_weights)
        model = ShallowModel(self.static_data, path_weights, params=experiment_params, online=False, train=True,
                             n_jobs=self.n_jobs)
        acc = model.fit(cv_masks)
        trial.set_user_attr('best_mae_test', model.best_mae_test)

        del model
        gc.collect()

        if isinstance(acc, np.ndarray):
            acc = float(np.mean(acc))

        return acc


def get_param_names(static_data, method):
    param_names = []
    for param, value in static_data[method].items():
        if isinstance(value, set):
            if len(value) > 1:
                param_names.append(param)
        elif isinstance(value, list):
            if len(value) > 1:
                param_names.append(param)

    if 'bootstrap_type' in static_data[method].items():
        if static_data[method]["bootstrap_type"] == "Bayesian":
            param_names.append("bagging_temperature")
        elif static_data[method]["bootstrap_type"] == "Bernoulli":
            param_names.append("subsample")
    return param_names


def find_init_params(static_data, method):
    if method == 'RF':
        model = RandomForestRegressor()
    elif method == 'CatBoost':
        model = CatBoostRegressor()
    elif method == 'lasso':
        if static_data['horizon_type'] == 'multi-output':
            model = MultiTaskLassoCV(max_iter=150000)
        else:
            model = LassoCV(max_iter=150000)
    else:
        raise ValueError(f'Unknown method {method} for shallow models')
    param_names = get_param_names(static_data, method)
    return {param: value for param, value in model.get_params().items() if param in param_names}


def optuna_thread(project_id, static_data, cluster_name, cluster_dir, method, refit=False):
    path_group = static_data['path_group']

    if not os.path.exists(os.path.join(cluster_dir, f'results_{cluster_name}_{method}.csv')) or refit:
        n_jobs = find_free_cpus(path_group)
        print(f'CPU methods starts running on {n_jobs} cpus')
        print(f'{method} Model of {cluster_name} of {project_id} is starts.....')
        if not os.path.exists(os.path.join(cluster_dir, f'study_{method}.pickle')):
            study = optuna.create_study(sampler=TPESampler(seed=42, consider_magic_clip=True, n_startup_trials=4,
                                                           n_ei_candidates=4))
            joblib.dump(study, os.path.join(cluster_dir, f'study_{method}.pickle'))
        else:
            try:
                study = joblib.load(os.path.join(cluster_dir, f'study_{method}.pickle'))
            except:
                study = optuna.create_study(sampler=TPESampler(seed=42, consider_magic_clip=True, n_startup_trials=4,
                                                               n_ei_candidates=4))

        # study.enqueue_trial(find_init_params(static_data, method))
        study.optimize(Objective(static_data, cluster_name, cluster_dir, method, n_jobs),
                       n_trials=static_data[method]['n_trials'],
                       gc_after_trial=True)
        results = study.trials_dataframe().sort_values(by='value')
        results.to_csv(os.path.join(cluster_dir, f'results_{cluster_name}_{method}.csv'))



def CPU_thread(static_data, method, cluster=None, refit=False):
    if cluster is not None:
        clusters = cluster
    else:
        clusters = joblib.load(os.path.join(static_data['path_model'], 'clusters.pickle'))
    for cluster_name, cluster_dir in clusters.items():
        try:
            optuna_thread(static_data['_id'], static_data, cluster_name, cluster_dir, method, refit=refit)
        except Exception as e:
            tb = traceback.format_exception(e)
            print("".join(tb))
            send_predictions(" ".join(tb))
            raise e


def train_clusters_on_cpus(static_data, cluster=None, method=None, refit=False):
    print('cpu')
    time.sleep(10)
    methods = []
    if method is None:
        for m, values in static_data['methods_cpu'].items():
            if 'RBF' not in m and values:
                methods.append(m)

    if method is None and cluster is None:
        for method in methods:
            CPU_thread(static_data, method, refit=refit)
            print(f'Training of {method} ends successfully')
    else:
        if method is not None and cluster is None:
            CPU_thread(static_data, method, refit=refit)
            print(f'Training of {method} ends successfully')
        elif method is None and cluster is not None:
            for method in methods:
                CPU_thread(static_data, method, cluster=cluster, refit=refit)
                print(f'Training of {method} ends successfully')
        else:
            CPU_thread(static_data, method, cluster=cluster, refit=refit)
