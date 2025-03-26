import os
import sys
path_pycharm = (os.path.normpath(os.path.normpath(os.path.dirname(__file__)))).split(os.sep)
path_pycharm = os.path.join(*path_pycharm[:-3])
if sys.platform == 'linux':
    path_pycharm = '/' + path_pycharm
print(path_pycharm)
sys.path.append(path_pycharm)

import time
import gc
import joblib
import copy

import numpy as np
import pandas as pd

from GPUtil import getGPUs
from eforecast.training.optimizers.optimizer_hyper import HyperoptOptimizer

from eforecast.common_utils.devops_utils import init_conda_in_exe
from eforecast.common_utils.devops_utils import run_exe
from eforecast.datasets.data_feeder import DataFeeder

class ObjectiveAbstractClass:

    def __init__(self, static_data, runtime_file, cluster_name, cluster_dir, method, refit=False):
        self.space_structure = None
        self.static_data = static_data
        self.method = method
        self.cluster_dir = cluster_dir
        self.cluster_name = cluster_name
        self.refit = refit
        self.runtime_file = runtime_file
        self.warming = self.static_data[method]['warming_iterations']
        self.use_image = self.static_data['use_image']
        self.nwp_data_merge = self.static_data['nwp_data_merge']
        self.nwp_data_compress = self.static_data['compress_data']
        self.scale_nwp_methods = self.static_data['scale_nwp_method']
        self.scale_row_methods = self.static_data['scale_row_method']
        self.feature_selection_methods = self.static_data['feature_selection_methods']
        self.data_feeder = DataFeeder(self.static_data, online=False, train=True)
        self.initialize(refit)

    def initialize(self, refit=False):
        self.define_space()
        self.define_structure_space()

    def get_param(self, name, type, dtype, space=None, range=None, values=None):
        return {name: {'type': type,
                       'dtype': dtype,
                       'space': space,
                       'range': range,
                       'values': values}}

    def select_structure(self, trial_structure, experiment_tag, exp):
        exp_sel = dict()
        for key, layers in exp.items():
            exp_sel[key] = []
            for i, layer in enumerate(layers):
                layer_type = layer[0]
                param = f'{experiment_tag}_{key}_{layer_type}_{i}'
                size = trial_structure[param] if param in trial_structure.keys() \
                    else self.fix_params_structure[experiment_tag][param]
                if isinstance(size, str):
                    if size not in 'linear':
                        size = float(size)
                elif isinstance(size, int):
                    size = float(size)
                exp_sel[key].append((layer_type, size))
        return exp_sel

    def define_structure_space(self):
        experiment_tags = list(self.static_data[self.method]['experiment_tag'])
        self.space_structure = dict()
        self.fix_params_structure = dict()
        self.param_layer_names = dict()
        for experiment_tag in experiment_tags:
            exp = self.static_data['experiments'][experiment_tag]
            self.space_structure[experiment_tag] = dict()
            self.fix_params_structure[experiment_tag] = dict()
            self.param_layer_names[experiment_tag] = []
            for key, layers in exp.items():
                for i, layer in enumerate(layers):
                    layer_type = layer[0]
                    sizes = layer[1]
                    param = f'{experiment_tag}_{key}_{layer_type}_{i}'
                    if isinstance(sizes, list):
                        if len(sizes) > 1:
                            self.space_structure[experiment_tag].update(
                                self.get_param(param, 'real', 'float', range=list(sizes)))
                        elif len(sizes) == 1:
                            self.fix_params_structure[experiment_tag][param] = list(sizes)[0]
                        else:
                            self.fix_params_structure[experiment_tag][param] = sizes
                    elif not isinstance(sizes, list) and not isinstance(sizes, set):
                        self.fix_params_structure[experiment_tag][param] = sizes
                    else:
                        if len(sizes) > 1:
                            self.space_structure[experiment_tag].update(
                                self.get_param(param, 'cat', 'float', values=list(sizes)))
                        elif len(sizes) == 1:
                            self.fix_params_structure[experiment_tag][param] = list(sizes)[0]
                        else:
                            self.fix_params_structure[experiment_tag][param] = sizes
        self.param_structure_names = dict()
        for experiment_tag, experiment in self.space_structure.items():
            self.param_structure_names[experiment_tag] = []
            for param_name, param_attr in experiment.items():
                self.param_structure_names[experiment_tag].append(param_name)

    def define_space(self):
        self.space = dict()
        self.fix_params = dict()
        params_method = copy.deepcopy(self.static_data[self.method])
        for param, value in params_method.items():
            if isinstance(value, set):
                if len(value) > 1:
                    if isinstance(list(value)[0], str):
                        self.space.update(self.get_param(param, 'cat', 'string', values=list(value)))
                    elif isinstance(list(value)[0], bool):
                        self.space.update(self.get_param(param, 'bool', 'bool'))
                    elif isinstance(list(value)[0], int):
                        self.space.update(self.get_param(param, 'cat', 'int', values=list(value)))
                    else:
                        self.space.update(self.get_param(param, 'cat', 'float', values=list(value)))
                else:
                    self.fix_params[param] = list(value)[0]
            elif isinstance(value, list):
                if len(value) > 1:
                    if isinstance(value[0], bool):
                        self.space.update(self.get_param(param, 'bool', 'bool'))
                    elif isinstance(value[0], int):
                        self.space.update(self.get_param(param, 'int', 'int', range=value))
                    else:
                        self.space.update(self.get_param(param, 'real', 'float', range=value))
                else:
                    self.fix_params[param] = value[0]
            else:
                self.fix_params[param] = value
        self.param_names = []
        for param_name, param_attr in self.space.items():
            self.param_names.append(param_name)

    def if_trial_exists(self, X_trials, trial):
        X_trial_check = []
        param_dict = dict()
        for key in self.param_names:
            param_dict[key] = trial[key]
        X_trial_check.append(param_dict)
        X_trial_check = pd.DataFrame(X_trial_check)
        for i in range(X_trials.shape[0]):
            check = (X_trial_check == X_trials.loc[i]).all(1)
            if isinstance(check, pd.Series):
                check = check.values[0]
            if check:
                return check
        if isinstance(check, pd.Series):
            check = check.values[0]
        return check

    def search_trial(self, X_trial, trial, where):
        check_trial = self.if_trial_exists(X_trial, trial)
        trial_new = copy.deepcopy(trial)
        if check_trial:
            trial_list = where
            for trial_ in trial_list:
                check_trial = self.if_trial_exists(X_trial, trial_)
                if not check_trial:
                    trial_new = copy.deepcopy(trial_)
                    break
        return trial_new

    def get_trials_df(self, trials):
        y_trial = []
        X_trial = []
        for trial in trials:
            param_dict = dict()
            for key in self.param_names:
                param_dict[key] = trial[key]
            X_trial.append(param_dict)
            y_trial.append(trial['value'])
        X_trial = pd.DataFrame(X_trial)
        return X_trial, y_trial

    def get_optim_trial(self, trials, fix_grid=False):
        n_trials = self.static_data[self.method]['n_trials']
        optimizer = HyperoptOptimizer(self.space)
        if len(trials) > 0:
            X_trial, y_trial = self.get_trials_df(trials)
            optimizer.observe(X_trial, np.array(y_trial))

        grid = self.grid_()
        ntr = len(grid)
        trial = optimizer.suggest(n_suggestions=1)[0]

        if len(trials) > 0:
            trial = self.search_trial(X_trial, trial,
                                      optimizer.suggest(n_suggestions=n_trials))
            if trial is None:
                trial = self.search_trial(X_trial, trial, grid)

            check_trial = self.if_trial_exists(X_trial, trial)
            if check_trial:
                trial_list = optimizer.suggest(n_suggestions=n_trials)
                trial = copy.deepcopy(trial_list[np.random.randint(n_trials - 1)])
        return trial

    def init_experiment_params(self, trial_number, trial, experiment_tag):
        image_in_data = False
        merge = trial['merge'] if 'merge' in trial.keys() else self.fix_params['merge']
        compress = trial['compress'] if 'compress' in trial.keys() else self.fix_params['compress']
        scale_row_method = trial['scale_row_method'] if 'scale' in trial.keys() else self.fix_params['scale_row_method']
        scale_nwp_method = trial['scale_nwp_method'] if 'scale' in trial.keys() else self.fix_params['scale_nwp_method']
        feature_selection_method = trial['feature_selection_method'] if 'feature_selection_method' in trial.keys() \
            else self.fix_params['feature_selection_method']
        data_types = dict()
        for data_tag in self.static_data['experiments'][experiment_tag].keys():
            if 'output' in data_tag or 'hidden' in data_tag:
                continue
            if 'row' in data_tag:
                if 'all' in data_tag or 'nwp' in data_tag:
                    data_types[data_tag] = {'scale_row_method': scale_row_method,
                                            'scale_nwp_method': scale_nwp_method,
                                            'merge': merge,
                                            'compress': compress,
                                            'feature_selection_method': feature_selection_method
                                            }
                else:
                    data_types[data_tag] = {'scale_row_method': scale_row_method,
                                            'feature_selection_method': feature_selection_method}
            elif 'lstm' in data_tag:
                data_types[data_tag] = {'scale_row_method': scale_row_method,
                                        'feature_selection_method': feature_selection_method
                                        }
            elif 'nwp' in data_tag:
                data_types[data_tag] = {'scale_nwp_method': scale_nwp_method,
                                        'merge': merge}
            elif 'image' in data_tag:
                image_in_data = True
                data_types[data_tag] = {}
        experiment_params = {'trial_number': trial_number,
                             'method': self.method,
                             'name': self.cluster_name,
                             'cluster_dir': self.cluster_dir,
                             'cluster': {'cluster_name':self.cluster_name,
                                         'cluster_path':self.cluster_dir},
                             'data_types': data_types,
                             'image_in_data': image_in_data,
                             'experiment_tag': experiment_tag}

        print(f'Run test {trial_number} with the following parameters')
        for param, value in trial.items():
            print(f'{param} has value {value}')
            if param not in experiment_params.keys():
                experiment_params[param] = value
        experiment_params.update(self.fix_params)

        return experiment_params

    def get_optim_structure(self, experiment_tag, trials):
        if len(self.space_structure[experiment_tag]) > 0:
            optimizer_structure = HyperoptOptimizer(self.space_structure[experiment_tag])
            if len(trials) > 0:
                y_trial_structure = []
                indices = []
                X_trial_structure = []
                for i, trial in enumerate(trials):
                    param_dict = dict()
                    for key in self.param_structure_names[experiment_tag]:
                        if key in trial.keys():
                            param_dict[key] = trial[key]
                    if len(param_dict) > 0:
                        indices.append(i)
                        X_trial_structure.append(param_dict)
                        y_trial_structure.append(trial['value'])
                if len(X_trial_structure) > 0:
                    X_trial_structure = pd.DataFrame(X_trial_structure)
                    optimizer_structure.observe(X_trial_structure, np.array(y_trial_structure))
            trial_structure = optimizer_structure.suggest(n_suggestions=1)[0]
        else:
            trial_structure = dict()
        return trial_structure

    def train_or_skip(self, trial_number, trial, trial_structure, path_weights, model,  params, gpu_id):
        acc = np.inf
        start = time.time()
        if model.is_trained:
            acc = model.best_mae_test
            for param in model.params.keys():
                if param in trial.keys():
                    trial[param] = model.params[param]
        else:
            while True:
                gpus = getGPUs()
                gpuUtil = gpus[gpu_id].load
                if gpuUtil < 0.9:
                    break
                else:
                    time.sleep(10)
            count_runs = 0
            while acc > self.static_data['max_performance'] and count_runs <= 0:
                print(f'Run test {trial_number} count_runs {count_runs}')
                try:
                    self._fit(trial_number, gpu_id)
                except:
                    pass
                count_runs += 1
                gpus = getGPUs()
                memory_util = gpus[gpu_id].memoryUtil
                try:
                    model.load()
                    acc = model.best_mae_test
                except:
                    acc = np.inf
                    pass
                if acc is None:
                    continue
                if acc > self.static_data['max_performance']:
                    continue
                elif not os.path.exists(os.path.join(path_weights, 'net_weights.pickle')):
                    print('Trial aboard due to gpu memory utilization')
                    print()
                    model.best_weights = {}
                    model.best_mae_test = np.inf
                    model.best_mae_val = np.inf
                    model.results = pd.DataFrame()
                    model.is_trained = True
                    model.save()
                    print('deep model failed')
                    break

                else:
                    break
        print(acc)
        params['value'] = acc
        params['mae_test'] = model.best_mae_test
        params['mae_val'] = model.best_mae_val
        params['sse_test'] = model.best_sse_test
        params['sse_val'] = model.best_sse_val
        params['duration'] = time.time() - start

        columns = ['trial_number', 'duration', 'value', 'mae_test', 'mae_val', 'sse_val', 'sse_test'] + self.param_names
        trial = {key: value for key, value in params.items() if key in columns}
        trial.update(trial_structure)

        if not np.isinf(acc):
            path_trials = os.path.join(self.cluster_dir, self.method, 'trials')
            file_trial = f'trial{trial_number}.pickle'
            joblib.dump(trial, os.path.join(path_trials, file_trial))
        del model
        gc.collect()

    def _fit(self, test_id, gpu_id):
        if sys.platform == 'linux':
            python_file = f"~/{self.static_data['env_name']}/bin/python"
        else:
            python_file = 'python'
        method = self.method
        path_weights = os.path.join(self.cluster_dir, method, f'test_{test_id}')
        exe_file_name = f"exe_{self.cluster_name}_{method}_gpu_{gpu_id}_test{test_id}.sh" if sys.platform == 'linux' \
            else f"exe_{self.cluster_name}_{method}_gpu_{gpu_id}_test{test_id}.bat"
        exe_file = os.path.join(path_weights, exe_file_name)
        if sys.platform == 'linux':
            path_pycharm = (os.path.normpath(os.path.dirname(__file__))).split(os.sep)
            path_pycharm = os.path.join(*path_pycharm[:-2])
            path_pycharm = '/' + path_pycharm
        else:
            path_pycharm = (os.path.normpath(os.path.dirname(__file__))).split(os.sep)
            path_pycharm = '/'.join(path_pycharm[:-2])
        init_conda_in_exe(self.static_data, path_pycharm, [exe_file])

        with open(exe_file, mode='a') as fp:
            fp.write(f"{python_file} {os.path.dirname(__file__)}/{self.runtime_file} "
                     f"{path_weights}  "
                     f"{gpu_id} "
                     f"{int(self.refit)}\n")
            file_weights = os.path.join(path_weights, 'net_weights.pickle')
            fp.write(f'if [ -f {file_weights} ]; then\n')
            fp.write(f'\techo "succeed"\n')
            fp.write('else\n')
            fp.write(f'\techo "failed" > {exe_file.replace(".sh", ".txt").replace(".bat", ".txt")}\n')
            fp.write('\texit\n')
            fp.write('fi\n')
        run_exe([exe_file])

    def grid_(self,):
        from sklearn.model_selection import ParameterGrid
        space_temp = dict()
        for key, value in self.space.items():
            if value['values'] is not None:
                space_temp[key] = value['values']
            elif value['range'] is not None:
                space_temp[key] = value['range']
            elif value['type'] == 'bool':
                space_temp[key] = [True, False]
            else:
                print('space param is not defined correctly')
                space_temp[key] = None
        grid = ParameterGrid(space_temp)
        return list(grid)
