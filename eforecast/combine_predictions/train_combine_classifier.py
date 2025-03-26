import os
import gc
import glob
import time
import shutil
import joblib
import optuna

import numpy as np
import pandas as pd

from optuna.samplers import TPESampler

from sklearn.model_selection import train_test_split

from eforecast.common_utils.dataset_utils import sync_datasets
from eforecast.common_utils.dataset_utils import sync_dict_df
from eforecast.common_utils.train_utils import find_free_cpus

from eforecast.datasets.data_feeder import DataFeeder
from eforecast.combine_predictions.shallow_classifier import ShallowModelClassifier

from catboost import CatBoostClassifier


class Objective(object):
    def __init__(self, static_data, cluster_name, combine_cluster_dir, cluster_dir, method, n_jobs, best_predictor,
                 predictors_id, refit=False):
        self.static_data = static_data
        self.method = method
        self.combine_cluster_dir = combine_cluster_dir
        self.cluster_dir = cluster_dir
        self.cluster_name = cluster_name
        self.n_jobs = n_jobs
        self.best_predictor = best_predictor
        self.predictors_id = predictors_id
        self.refit = refit

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
        self.static_data[self.method]["boosting_type"] = {"Plain"}
        self.static_data[self.method]["bootstrap_type"] = {"Bayesian", "Bernoulli"}
        for param, value in self.static_data[self.method].items():
            if param in {'eval_metric', 'objective'}:
                value = 'MultiClass'
            if param == 'depth':
                if self.static_data['horizon_type'] == 'multi-output':
                    value = [2, 5]
                else:
                    value = [2, 8]
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

        if 'bootstrap_type' in experiment_params.keys():
            if experiment_params["bootstrap_type"] == "Bayesian":
                experiment_params["bagging_temperature"] = 6
            elif experiment_params["bootstrap_type"] == "Bernoulli":
                experiment_params["subsample"] = 0.5
        path_weights = os.path.join(self.combine_cluster_dir, f'test_{trial.number}')
        if not os.path.exists(path_weights):
            os.makedirs(path_weights)
        else:
            shutil.rmtree(path_weights)
            os.makedirs(path_weights)
        model = ShallowModelClassifier(self.static_data, path_weights, predictors=self.predictors_id,
                                       params=experiment_params, online=False, train=True,
                                       n_jobs=self.n_jobs,
                                       refit=self.refit)
        acc = model.fit(self.best_predictor, cv_masks)
        trial.set_user_attr('best_mae_test', model.best_mae_test)

        del model
        gc.collect()
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
    if method == 'CatBoost':
        model = CatBoostClassifier()
    else:
        raise ValueError(f'Unknown method {method} for shallow models')
    param_names = get_param_names(static_data, method)
    return {param: value for param, value in model.get_params().items() if param in param_names}


def optuna_thread(project_id, static_data, cluster_name, combine_cluster_dir, cluster_dir, method, best_predictor,
                  predictors_id, refit=False):
    path_group = static_data['path_group']
    n_jobs = find_free_cpus(path_group)
    print(f'CPU methods starts running on {n_jobs} cpus')
    print(f'{method} Model of {cluster_name} of {project_id} is starts.....')

    if not os.path.exists(os.path.join(combine_cluster_dir, f'results_{cluster_name}_{method}.csv')) or refit:
        if not os.path.exists(os.path.join(combine_cluster_dir, f'study_{method}.pickle')):
            study = optuna.create_study(sampler=TPESampler(seed=42, consider_magic_clip=True, n_startup_trials=4,
                                                           n_ei_candidates=4), direction='maximize')
            joblib.dump(study, os.path.join(combine_cluster_dir, f'study_{method}.pickle'))
        else:
            try:
                study = joblib.load(os.path.join(combine_cluster_dir, f'study_{method}.pickle'))
            except:
                study = optuna.create_study(sampler=TPESampler(seed=42, consider_magic_clip=True, n_startup_trials=4,
                                                               n_ei_candidates=4), direction='maximize')

        study.optimize(Objective(static_data, cluster_name, combine_cluster_dir, cluster_dir, method, n_jobs,
                                 best_predictor, predictors_id, refit=refit),
                       n_trials=static_data[method]['n_trials'],
                       gc_after_trial=True)
        results = study.trials_dataframe().sort_values(by='value', ascending=False)
        model_dir = combine_cluster_dir
        test_dir = os.path.join(combine_cluster_dir, f'test_{study.best_trial.number}')
        for filename in glob.glob(os.path.join(test_dir, '*.*')):
            print(filename)
            shutil.copy(filename, model_dir)
        results.to_csv(os.path.join(combine_cluster_dir, f'results_{cluster_name}_{method}.csv'))
        for number in range(len(study.get_trials())):
            test_dir = os.path.join(combine_cluster_dir, f'test_{number}')
            if os.path.exists(test_dir):
                shutil.rmtree(test_dir)


def train_classifier(static_data, method, cluster_name, combine_cluster_dir, cluster_dir, best_predictor,
                     predictors_id, refit=False):
    optuna_thread(static_data['_id'], static_data, cluster_name, combine_cluster_dir, cluster_dir, method,
                  best_predictor, predictors_id, refit=refit)
