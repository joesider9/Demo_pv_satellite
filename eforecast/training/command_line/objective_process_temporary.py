import os
import sys
import traceback
if sys.platform == 'linux':
    path_pycharm = (os.path.normpath(os.path.dirname(__file__))).split(os.sep)
    path_pycharm = os.path.join(*path_pycharm[:-3])
    path_pycharm = '/' + path_pycharm
else:
    path_pycharm = (os.path.normpath(os.path.dirname(__file__))).split(os.sep)
    path_pycharm = '/'.join(path_pycharm[:-3])

print(path_pycharm)
sys.path.append(path_pycharm)
import shutil
import joblib
import pandas as pd

from eforecast.training.command_line.objective_abstract import ObjectiveAbstractClass
from eforecast.common_utils.dataset_utils import create_image_batches

try:
    import torch
    from eforecast.deep_models.pytorch_2x.network import DeepNetwork
    print('Backend is set pytorch')
except:
    try:
        from eforecast.deep_models.tf_2x.network import DeepNetwork
        from eforecast.deep_models.tf_2x.transformers.tranformer_network import TransformerNetwork
        print('Backend is set Tensorflow 2.10')
    except:
        raise ImportError('Cannot find backend')


# if not sys.argv[1].startswith('tl'):
#     trial_number = int(sys.argv[1])
#     method = sys.argv[2]
#     cluster_name = sys.argv[3]
#     path_cluster = sys.argv[4]
#     gpu_id = int(sys.argv[5])
#     refit = bool(int(sys.argv[6]))
#     print(f'Get trial_number: {trial_number}, method: {method}, cluster_name: {cluster_name},\n '
#           f'path_cluster: {path_cluster},\n gpu_id: {gpu_id}, refit: {refit}')
# else:
#     trial_number = sys.argv[1]
#     method = sys.argv[2]
#     cluster_name = sys.argv[3]
#     path_cluster = sys.argv[4]
#     tl_path_weights = sys.argv[5]
#     tl_cluster_name = sys.argv[6]
#     tl_cluster_dir = sys.argv[7]
#     gpu_id = int(sys.argv[8])
#     refit = bool(int(sys.argv[9]))
#     print(f'Get trial_number: {trial_number}, method: {method}, cluster_name: {cluster_name}\n, '
#           f'path_cluster: {path_cluster},\n '
#           f'tl_path_weights: {tl_path_weights},\n '
#           f'tl_cluster_name: {tl_cluster_name},\n '
#           f'tl_cluster_dir: {tl_cluster_dir},\n '
#           f'gpu_id: {gpu_id}, refit: {refit}')

trial_number = 0
method = 'LSTM'
cluster_name = 'global'
path_cluster =  ('/media/smartrue/HHD1/George/models/PPC/PPC_sat_ver2/pv/site_/multi-output/model_ver0/global')
gpu_id = 0
refit = bool(0)
#
tl_path_weights = ''
tl_cluster_name = ''
tl_cluster_dir = ''


class ObjectiveProcess(ObjectiveAbstractClass):
    def __init__(self, static_data, runtime_file, cluster_name,
                 cluster_dir, method, refit=False):
        super(ObjectiveProcess, self).__init__(static_data, runtime_file, cluster_name,
                                               cluster_dir, method, refit)

    def fit_trial(self, trial_number, trials, gpu_i, fix_grid=False):
        print(f'Objective process starts for {self.cluster_name} to train trial'
              f' {trial_number} of model {self.method} ')
        print('\n')

        if self.static_data[method].get('fix_grid_search', False):
            file_grid = os.path.join(self.cluster_dir, self.method,'trial_df.csv')
            if os.path.exists(file_grid):
                grid = pd.read_csv(file_grid, header=0).to_dict('records')
            else:
                grid = self.grid_()
                pd.DataFrame(grid).to_csv(file_grid, index=False)
            trial = grid[trial_number]
        else:
            trial = self.get_optim_trial(trials)

        experiment_tag = trial['experiment_tag'] if 'experiment_tag' in trial.keys() \
            else self.fix_params['experiment_tag']

        experiment_params = self.init_experiment_params(trial_number, trial, experiment_tag)

        trial_structure = self.get_optim_structure(experiment_tag, trials)

        experiment_params['experiment'] = self.select_structure(trial_structure, experiment_tag,
                                                                self.static_data['experiments'][experiment_tag])
        experiment_params['inputs'] = [key for key in experiment_params['experiment'].keys() if 'output' not in key]

        path_weights = os.path.join(self.cluster_dir,
                                    self.method,
                                    f'test_{trial_number}')
        if 'images' in experiment_params['experiment'] and not experiment_params.get('create_image_dataset_real_time',
                                                                                     False):
            use_target = self.static_data['type'] == 'image2image'
            experiment_params = create_image_batches(self.static_data, experiment_params,
                                                     experiment_params['batch_size'], gpu_id, use_target)

        if os.path.exists(path_weights) and self.refit:
            shutil.rmtree(path_weights)
        if not os.path.exists(path_weights):
            os.makedirs(path_weights)
        joblib.dump(experiment_params, os.path.join(path_weights, 'parameters.pickle'))

        model = DeepNetwork(self.static_data, path_weights, experiment_params, refit=self.refit)
        self.train_or_skip(trial_number, trial, trial_structure, path_weights, model, experiment_params, gpu_id)

    def fit_transfer_learning(self,):
        path_weights = os.path.join(self.cluster_dir,
                                    self.method,
                                    f'test_{trial_number}')

        if os.path.exists(path_weights) and self.refit:
            shutil.rmtree(path_weights)
        if not os.path.exists(path_weights):
            os.makedirs(path_weights)

        experiment_params = joblib.load(os.path.join(tl_path_weights, 'parameters.pickle'))

        trial = dict()
        for p in experiment_params.keys():
            if p in self.param_names:
                trial[p] = experiment_params[p]

        experiment_params.update({'trial_number': trial_number,
                                  'name': self.cluster_name,
                                  'cluster_dir': self.cluster_dir,
                                  'cluster': {'cluster_name': self.cluster_name,
                                              'cluster_path':self.cluster_dir},
                                  'tl_path_weights': tl_path_weights})
        joblib.dump(experiment_params, os.path.join(path_weights, 'parameters.pickle'))

        model = DeepNetwork(self.static_data, path_weights, experiment_params, train=True, refit=self.refit)
        trial_structure = dict()
        self.train_or_skip(trial_number, trial, trial_structure, path_weights, model, experiment_params, gpu_id)


if __name__ == '__main__':
    static_data = joblib.load(os.path.join(path_cluster, 'static_data.pickle'))
    # from bittakos.short_term_image.configuration.config import config
    # from eforecast.init.initialize import initializer
    #
    # static_data = initializer(config())
    path_trials = os.path.join(path_cluster, method, 'trials')
    if not os.path.exists(path_trials):
        os.makedirs(path_trials)
    trials = []
    for trial in sorted(os.listdir(path_trials)):
        trials.append(joblib.load(os.path.join(path_trials, trial)))
    objective = ObjectiveProcess(static_data, 'run_net_on_gpu.py', cluster_name, path_cluster, method,
                                 refit=refit)
    try:
        if not str(trial_number).startswith('tl'):
            fix_grid = False
            objective.fit_trial(trial_number, trials, gpu_id, fix_grid=fix_grid)
        else:
            objective.fit_transfer_learning()
    except Exception as e:
        tb = traceback.format_exception(e)
        print("".join(tb))
        with open(os.path.join(os.path.join(path_cluster, method), 'error.txt'), mode='w') as fp:
            fp.write(" ".join(tb))
        raise Exception("".join(tb))


