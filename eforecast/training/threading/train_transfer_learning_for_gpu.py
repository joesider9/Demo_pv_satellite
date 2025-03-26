import os
import gc
import shutil
import joblib
from joblib import Parallel
from joblib import delayed
import numpy as np
import pandas as pd

from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed

from eforecast.common_utils.train_utils import distance
from eforecast.training.threading.objective_abstract import ObjectiveAbstractClass


try:
    import torch
    from eforecast.deep_models.pytorch_2x.network import DeepNetwork
    print('Backend is set pytorch')
except:
    try:
        import tensorflow as tf

        from eforecast.deep_models.tf_2x.network import DeepNetwork
        from eforecast.deep_models.tf_2x.transformers.tranformer_network import TransformerNetwork
        print('Backend is set Tensorflow 2.10')
    except:
        raise ImportError('Cannot find backend')

class TransferLearner(ObjectiveAbstractClass):
    def __init__(self, static_data, cluster_name, cluster_dir, method, refit=False):
        super(TransferLearner, self).__init__(static_data, cluster_name,
                                        cluster_dir, method, refit)

    def fit(self, exp_name, cluster, tl_path_weights, gpu_i):
        cv_masks = joblib.load(os.path.join(self.cluster_dir, 'cv_mask.pickle'))
        trial_number = f'tl{exp_name.split("_")[-1]}'
        path_weights = os.path.join(self.cluster_dir,
                                    self.method,
                                    '_tl'.join(exp_name.split('_')))

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
                                 'cluster': cluster,
                                  'tl_path_weights': tl_path_weights})
        joblib.dump(experiment_params, os.path.join(path_weights, 'parameters.pickle'))

        model = DeepNetwork(self.static_data, path_weights, experiment_params, train=True, refit=self.refit)

        params = self.train_or_skip(trial_number, trial, path_weights, model,
                                    cv_masks,  experiment_params, gpu_i)
        if np.isinf(params['value']):
            model = DeepNetwork(self.static_data, path_weights, experiment_params, train=True, refit=True)

            params = self.train_or_skip(trial_number, trial, path_weights, model,
                                        cv_masks, experiment_params, gpu_i)
        columns = ['trial_number', 'duration', 'value', 'mae_test', 'mae_val', 'sse_val', 'sse_test'] + self.param_names
        trial = {key: value for key, value in params.items() if key in columns}
        del model
        gc.collect()
        return trial


def get_results(static_data, cluster_dir, trial, method):
    path_weights = os.path.join(cluster_dir, method, f'test_{trial}')
    model = DeepNetwork(static_data, path_weights)

    return model.results


def run_transfer_learning(project_id, static_data, cluster_name, cluster_dir, cluster_base_dir, method, n_jobs,
                          refit, gpu_i):
    print(f'{method} Model of {cluster_name} of {project_id} is starts.....')
    if not os.path.exists(os.path.join(cluster_dir, f'results_{cluster_name}_{method}.csv')) or refit:

        transfer_learner = TransferLearner(static_data, cluster_name, cluster_dir, method, refit=refit)
        cluster_base_method_dir = os.path.join(cluster_base_dir, method)
        cluster = {'cluster_name': cluster_name,
                      'cluster_path': cluster_dir}
        with Parallel(n_jobs=n_jobs, prefer='threads') as parallel:
            trials = parallel(delayed(transfer_learner.fit)(exp_name, cluster,
                                                               os.path.join(cluster_base_method_dir, exp_name),
                                                               gpu_i)
                                 for exp_name in os.listdir(cluster_base_method_dir) if exp_name.startswith('test'))
        trials = pd.DataFrame(trials)

        results = trials.sort_values(by='value')
        cols = ['mae_test', 'mae_val',
                'sse_test', 'sse_val']
        res = results[cols]
        res = res.clip(1e-6, 1e6)
        diff_mae = pd.DataFrame(np.abs(res['mae_test'].values - res['mae_val'].values),
                                index=res.index, columns=['diff_mae'])
        res = pd.concat([res, diff_mae], axis=1)
        diff_sse = pd.DataFrame(np.abs(res['sse_test'].values - res['sse_val'].values), index=res.index,
                                columns=['diff_sse'])
        res = pd.concat([res, diff_sse], axis=1)
        res_old, res_max, res_min = 1000 * np.ones(6), 1000 * np.ones(6), 1000 * np.ones(6)
        i = 0
        best_trials = []
        weights = np.array([0.5, 0.5, 0.1, 0.1, 0.01, 0.01])
        while res.shape[0] > 0:
            flag_res, res_old, res_max, res_min = distance(res.iloc[i].values, res_old, res_max, res_min,
                                                           weights=weights)
            if flag_res:
                best = i
            i += 1
            if i == res.shape[0]:
                best_trials.append(res.index[best])
                i = 0
                res_old, res_max, res_min = 1000 * np.ones(6), 1000 * np.ones(6), 1000 * np.ones(6)
                res = res.drop(index=res.index[best])
        results = results.loc[best_trials]
        results.to_csv(os.path.join(cluster_dir, f'results_{cluster_name}_{method}.csv'))
    else:
        results = pd.read_csv(os.path.join(cluster_dir, f'results_{cluster_name}_{method}.csv'), index_col=0, header=0)
    return results.trial_number.values[0]


def GPU_thread(static_data, static_data_base, n_gpus, n_jobs, cluster=None, method=None, refit=False):
    clusters = joblib.load(os.path.join(static_data['path_model'], 'clusters.pickle'))

    res = []
    gpu_ids = {cluster_name: i % n_gpus for i, cluster_name in enumerate(clusters.keys())}
    clusters_base = joblib.load(os.path.join(static_data_base['path_model'], 'clusters.pickle'))
    global_dir_base = os.path.join(static_data_base['path_model'], 'global')
    clusters_base.update({'global': global_dir_base})
    with ThreadPoolExecutor(max_workers=n_gpus) as executor:
        futures = [executor.submit(run_transfer_learning, static_data['_id'],
                                   static_data, cluster_name, cluster_dir, clusters_base[cluster_name],
                                   method,
                                   n_jobs, refit, gpu_ids[cluster_name])
                   for cluster_name, cluster_dir in clusters.items()]
        for future in as_completed(futures):
            res.append(future.result())

    best_trials = dict()
    i = 0
    for cluster_name, cluster_dir in clusters.items():
        best_trials[cluster_name] = dict()
        best_trials[cluster_name]['best'] = res[i]
        best_trials[cluster_name]['path'] = cluster_dir
        i += 1

    return best_trials


def train_transfer_learning_on_gpus(static_data, static_data_base, refit=False):
    print('Train Deep learning models on gpu')
    cluster_methods = [method for method, values in static_data['cluster_methods'].items() if values]

    global_methods = [method for method, values in static_data['global_methods'].items() if values]

    n_gpus = static_data['n_gpus']

    for method in global_methods:
        global_dir = os.path.join(static_data['path_model'], 'global')
        if not os.path.exists(global_dir):
            os.makedirs(global_dir)
        clusters = {'global': global_dir}
        GPU_thread(static_data, n_gpus, static_data[method]['n_jobs'],
                   cluster=clusters, method=method, refit=refit)
        print(f'Training of {method} ends successfully')

    if static_data['is_Fuzzy']:
        clusters = joblib.load(os.path.join(static_data['path_model'], 'clusters.pickle'))
        for method in cluster_methods:
            GPU_thread(static_data, static_data_base, n_gpus, static_data[method]['n_jobs'],
                   cluster=clusters, method=method, refit=refit)
            print(f'Training of {method} ends successfully')
