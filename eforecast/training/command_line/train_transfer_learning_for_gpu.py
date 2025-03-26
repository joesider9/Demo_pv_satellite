import os
import joblib

import numpy as np

from eforecast.common_utils.devops_utils import init_exe_files
from eforecast.common_utils.devops_utils import run_tasks
from eforecast.common_utils.devops_utils import get_results
from eforecast.common_utils.devops_utils import check_if_exists
from eforecast.common_utils.devops_utils import save_deep_models
from eforecast.common_utils.devops_utils import is_cuda_oom
from eforecast.training.command_line.objective_process_temporary import trial_number


def check_if_refit(path_weights, refit, file_trial=None):
    if os.path.exists(os.path.join(path_weights, 'error.txt')):
        error = ' '
        with open(os.path.join(path_weights, 'error.txt'), 'r') as f:
            error = f.read()
            f.close()
        cuda_err = is_cuda_oom(error)
        if not cuda_err:
            raise ValueError(f'Runtime error occurred in experiment {path_weights}')
    else:
        cuda_err = False

    if refit or (os.path.exists(os.path.join(path_weights, 'error.txt')) and cuda_err):
        refit_trial = 1
    else:
        if file_trial is not None:
            trial = joblib.load(file_trial)
            if np.isnan(trial['value']) or np.isinf(trial['value']):
                refit_trial = 1
            else:
                refit_trial = 0
        else:
            refit_trial = int(refit)
    return refit_trial

def GPU_thread(static_data, n_gpus, n_jobs, method, clusters, runtime_file, static_data_base, refit=False):
    project_id = static_data['_id']
    exe_files = init_exe_files(static_data, n_gpus, n_jobs, method)
    tasks = []
    i = 0
    j = 0
    clusters_base = joblib.load(os.path.join(static_data_base['path_model'], 'clusters.pickle'))
    global_dir_base = os.path.join(static_data_base['path_model'], 'global')
    clusters_base.update({'global': global_dir_base})
    for cluster_name, cluster_dir in clusters.items():
        print(f'{method} Model of {cluster_name} of {project_id} is starts.....')
        if not os.path.exists(os.path.join(cluster_dir, f'results_{cluster_name}_{method}.csv')) or refit:
            joblib.dump(static_data, os.path.join(cluster_dir, 'static_data.pickle'))
            for experiment in os.listdir(os.path.join(clusters_base[cluster_name], method)):
                if not os.path.isdir(os.path.join(clusters_base[cluster_name], method, experiment)):
                    continue
                if experiment == 'trials':
                    continue
                trial_number = experiment.split('_')[-1]
                path_weights = os.path.join(cluster_dir, method, experiment)
                file_trial = os.path.join(cluster_dir, method, 'trials', f'trial{trial_number}.pickle')
                if os.path.exists(file_trial):
                    refit_trial = check_if_refit(path_weights, refit, file_trial=file_trial)
                else:
                    refit_trial = 1 if refit else 0

                tl_path_weights = os.path.join(clusters_base[cluster_name], method, experiment)

                task = {'trial_number': trial_number, 'method': method,
                        'cluster_name': cluster_name, 'cluster_dir': cluster_dir,
                        'tl_path_weights': tl_path_weights,
                        'tl_cluster_name': cluster_name,
                        'tl_cluster_dir': clusters_base[cluster_name],
                        'gpu_id': i % n_gpus,
                        'job_id': j % n_jobs,
                        'refit': refit_trial}
                tasks.append(task)
                j += 1
                if j % n_jobs == 0:
                    i += 1

    if len(tasks) > 0:
        run_tasks(static_data, runtime_file, tasks, exe_files)
    best_trials = get_results(clusters, method, refit)



def train_transfer_learning_on_gpus(static_data, static_data_base, refit=False):
    print('Train Deep learning models on gpu')
    cluster_methods = [method for method, values in static_data['cluster_methods'].items() if values]
    global_methods = [method for method, values in static_data['global_methods'].items() if values]

    n_gpus = static_data['n_gpus']

    for method in global_methods:
        runtime_file = 'objective_process.py'
        global_dir = os.path.join(static_data['path_model'], 'global')
        if not os.path.exists(global_dir):
            os.makedirs(global_dir)
        clusters = {'global': global_dir}
        GPU_thread(static_data, n_gpus, static_data[method]['n_jobs'], method,
                   clusters, runtime_file, static_data_base, refit=refit)
        print(f'Training of {method} ends successfully')

    if static_data['is_Fuzzy']:
        clusters = joblib.load(os.path.join(static_data['path_model'], 'clusters.pickle'))
        for method in cluster_methods:
            runtime_file = 'objective_process.py'
            GPU_thread(static_data, n_gpus, static_data[method]['n_jobs'], method,
                       clusters, runtime_file, static_data_base, refit=refit)
            print(f'Training of {method} ends successfully')


