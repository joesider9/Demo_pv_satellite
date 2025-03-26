import os
import sys
import gc
import glob
import time
import shutil
import joblib

import numpy as np
import pandas as pd

from subprocess import Popen, PIPE, STDOUT

from eforecast.common_utils.train_utils import distance
import torch

def is_cuda_oom(exception: str) -> bool:
    """
    Checks if `exception` relates to CUDA out-of-memory, XPU out-of-memory, CUDNN not supported, or CPU out-of-memory

    Args:
        exception (`Exception`):
            An exception
    """
    _statements = [
        "CUDA out of memory.",  # CUDA OOM
        "XPU out of memory.",  # XPU OOM
        "cuDNN error: CUDNN_STATUS_NOT_SUPPORTED.",  # CUDNN SNAFU
        "DefaultCPUAllocator: can't allocate memory",  # CPU OOM
    ]
    return any(err in exception for err in _statements)


def make_executable(path):
    mode = os.stat(path).st_mode
    mode |= (mode & 0o444) >> 2  # copy R bits to X
    os.chmod(path, mode)


def init_conda_in_exe(static_data, path_pycharm, exe_files):
    for exe_file in exe_files:
        with open(exe_file, mode='w') as fp:
            if sys.platform == 'linux':
                fp.write('#!/bin/bash\n')
                fp.write('source ~/.bashrc\n')
                fp.write('source /etc/environment\n')
                fp.write(f"source {static_data['path_env']}/activate\n")
                # fp.write(f"conda activate {static_data['env_name']}\n")
            else:
                fp.write(f"set root={static_data['path_env']}/{static_data['env_name']}\n")
                fp.write(f"call {static_data['path_env']}/Scripts/activate {static_data['env_name']}\n")
                if os.path.exists('D:/'):
                    fp.write(f"d:\n")
            fp.write(f"cd {path_pycharm}\n")

def init_exe_files(static_data, n_gpus, n_jobs, method):
    if not os.path.exists(os.path.join(static_data['path_model'], 'exe_files')):
        os.makedirs(os.path.join(static_data['path_model'], 'exe_files'))
    exe_files = set()
    for gpu_id in range(n_gpus):
        for job_id in range(n_jobs):
            exe_file_name = f'exe_{method}_gpu_{gpu_id}_job{job_id}.sh' if sys.platform == 'linux' \
                                                                    else f"exe_{method}_gpu_{gpu_id}_job{job_id}.bat"
            exe_file = os.path.join(static_data['path_model'], 'exe_files', exe_file_name)
            exe_files.add(exe_file)

    if sys.platform == 'linux':
        path_pycharm = (os.path.normpath(os.path.dirname(__file__))).split(os.sep)
        path_pycharm = os.path.join(*path_pycharm[:-2])
        path_pycharm = '/' + path_pycharm
    else:
        path_pycharm = (os.path.normpath(os.path.dirname(__file__))).split(os.sep)
        path_pycharm = '/'.join(path_pycharm[:-2])
    init_conda_in_exe(static_data, path_pycharm, exe_files)
    return exe_files


def run_tasks(static_data,  runtime_file, tasks, exe_files):
    if sys.platform == 'linux':
        python_file = f"~/{static_data['env_name']}/bin/python"
    else:
        python_file = 'python'

    if sys.platform == 'linux':
        path_pycharm = (os.path.normpath(os.path.dirname(__file__))).split(os.sep)
        path_pycharm = os.path.join(*path_pycharm[:-2])
        path_pycharm = '/' + path_pycharm
    else:
        path_pycharm = (os.path.normpath(os.path.dirname(__file__))).split(os.sep)
        path_pycharm = '/'.join(path_pycharm[:-2])

    for task in tasks:
        gpu_id, job_id = task['gpu_id'], task['job_id']
        exe_file_name = f"exe_{task['method']}_gpu_{gpu_id}_job{job_id}.sh" if sys.platform == 'linux' \
            else f"exe_{task['method']}_gpu_{gpu_id}_job{job_id}.bat"
        exe_file = os.path.join(static_data['path_model'], 'exe_files', exe_file_name)
        if not exe_file in exe_files:
            raise ValueError('Cannot find exefile')
        with open(exe_file, mode='a') as fp:
            path_objective = os.path.join(path_pycharm, 'eforecast', 'training', 'command_line')
            if str(task['trial_number']).startswith('tl'):
                fp.write(f"{python_file} {path_objective}/{runtime_file} "
                         f"{task['trial_number']}  "
                         f"{task['method']}  "
                         f"{task['cluster_name']}  "
                         f"{task['cluster_dir']}  "
                         f"{task['tl_path_weights']}  "
                         f"{task['tl_cluster_name']}  "
                         f"{task['tl_cluster_dir']}  "
                         f"{gpu_id} "
                         f"{task['refit']}\n")
            else:
                fp.write(f"{python_file} {path_objective}/{runtime_file} "
                         f"{task['trial_number']}  "
                         f"{task['method']}  "
                         f"{task['cluster_name']}  "
                         f"{task['cluster_dir']}  "
                         f"{gpu_id} "
                         f"{task['refit']}\n")

            file_weights = os.path.join(task['cluster_dir'], task['method'], f"test_{task['trial_number']}", 'net_weights.pickle')
            fp.write(f'if [ -f {file_weights} ]; then\n')
            fp.write(f'\techo "succeed"\n')
            fp.write('else\n')
            fp.write(f'\techo "failed" > {exe_file.replace(".sh", ".txt").replace(".bat", ".txt")}\n')
            fp.write('fi\n')
    run_exe(exe_files)


def run_exe(exe_files):
    procs = []
    for exe_file in exe_files:
        if os.path.exists(exe_file.replace(".sh", ".txt").replace(".bat", ".txt")):
            os.remove(exe_file.replace(".sh", ".txt").replace(".bat", ".txt"))
        with open(exe_file, mode='a') as fp:
            fp.write(f'echo "Done" > {exe_file.replace(".sh", ".txt").replace(".bat", ".txt")}')
        make_executable(exe_file)
        if sys.platform == 'linux':
            procs.append(Popen(['gnome-terminal', '--title', os.path.basename(exe_file), '--', 'bash', '-c', exe_file],
                               stdin=PIPE, stdout=PIPE))
        else:
            procs.append(Popen([exe_file.replace('\\', '/')],
                               shell=True, stdin=PIPE, stdout=PIPE))
            os.system("title " + os.path.basename(exe_file))
    for proc in procs:
        proc.wait()
    while True:
        exists = []
        for exe_file in exe_files:
            exists.append(os.path.exists(exe_file.replace('.sh', '.txt').replace(".bat", ".txt")))
        if all(exists):
            done = []
            for exe_file in exe_files:
                with open(exe_file.replace('.sh', '.txt').replace(".bat", ".txt"), mode='r') as fp:
                    done.append(fp.read())
            if all(['Done' in d for d in done]):
                break
            elif any(['failed' in d for d in done]):
                raise RuntimeError('Some processes fail')
            else:
                raise RuntimeError('Unknown status')
        else:
            time.sleep(3)




def get_results(clusters, method, refit):
    best_models = dict()
    for cluster_name, cluster_dir in clusters.items():
        if not os.path.exists(os.path.join(cluster_dir, f'results_{cluster_name}_{method}.csv')) or refit:
            path_trials = os.path.join(cluster_dir, method, 'trials')
            if not os.path.exists(path_trials):
                continue
            shared_trials = []
            for trial in sorted(os.listdir(path_trials)):
                shared_trials.append(joblib.load(os.path.join(path_trials, trial)))
            trials = []
            for trial in shared_trials:
                param_dict = dict()
                for key in trial.keys():
                    param_dict[key] = trial[key]
                trials.append(param_dict)

            trials = pd.DataFrame(trials)
            results = trials.sort_values(by='value', ascending=False)
            cols = ['mae_test', 'mae_val',
                    'sse_test', 'sse_val']
            res = results[cols]
            res = res.clip(1e-6, 1e6)
            diff_mae = pd.DataFrame(np.abs(res['mae_test'].values - res['mae_val'].values),
                                    index=res.index, columns=['diff_mae'])
            res = pd.concat([res, 100 * diff_mae], axis=1)
            diff_sse = pd.DataFrame(np.abs(res['sse_test'].values - res['sse_val'].values), index=res.index,
                                    columns=['diff_sse'])
            res = pd.concat([res, diff_sse], axis=1)
            res_old, res_max, res_min = np.inf * np.ones(6), np.inf * np.ones(6), np.inf * np.ones(6)
            i = 0
            best_trials = []
            weights = np.array([0.5, 0.5, 0.25, 0.25, 0.25, 0.25])
            while res.shape[0] > 0:
                flag_res, res_old, res_max, res_min = distance(res.iloc[i].values, res_old, res_max, res_min,
                                                               weights=weights)
                if flag_res:
                    best = i
                i += 1
                if i == res.shape[0]:
                    best_trials.append(res.index[best])
                    i = 0
                    res_old, res_max, res_min = np.inf * np.ones(6), np.inf * np.ones(6), np.inf * np.ones(6)
                    res = res.drop(index=res.index[best])
            results = results.loc[best_trials]
            results.to_csv(os.path.join(cluster_dir, f'results_{cluster_name}_{method}.csv'))

            best_models[cluster_name] = dict()
            best_models[cluster_name]['best'] = results.trial_number.values[0]
            best_models[cluster_name]['path'] = cluster_dir
    return best_models


def check_if_exists(clusters, method):
    for cluster_name, cluster_dir in clusters.items():
        model_dir = os.path.join(cluster_dir, method)
        if not os.path.exists(os.path.join(model_dir, 'net_weights.pickle')):
            raise ImportError(f'check_if_exists: Cannot find model for {method} of cluster {cluster_name}')

def save_deep_models(results, method):
    for cluster_name, res in results.items():
        model_dir = os.path.join(res['path'], method)
        test_dir = os.path.join(model_dir, 'test_' + str(res['best']))
        for filename in glob.glob(os.path.join(test_dir, '*.*')):
            print(filename)
            shutil.copy(filename, model_dir)
        # for test_dir_name in os.listdir(model_dir):
        #     if 'test' in test_dir_name:
        #         test_dir = os.path.join(model_dir, test_dir_name)
        #         if os.path.exists(test_dir):
        #             shutil.rmtree(test_dir)

def gc_cuda():
    """Gargage collect Torch (CUDA) memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def get_cuda_total_memory():
    if torch.cuda.is_available():
        return torch.cuda.get_device_properties(0).total_memory
    return 0


def get_cuda_assumed_available_memory():
    if torch.cuda.is_available():
        return get_cuda_total_memory() - torch.cuda.memory_reserved()
    return 0


def get_cuda_available_memory():
    # Always allow for 1 GB overhead.
    if torch.cuda.is_available():
        return get_cuda_assumed_available_memory()
    return 0


def is_cuda_out_of_memory(exception):
    return (
        isinstance(exception, RuntimeError) and len(exception.args) == 1 and "CUDA out of memory." in exception.args[0]
    )


def is_cudnn_snafu(exception):
    # For/because of https://github.com/pytorch/pytorch/issues/4107
    return (
        isinstance(exception, RuntimeError)
        and len(exception.args) == 1
        and "cuDNN error: CUDNN_STATUS_NOT_SUPPORTED." in exception.args[0]
    )


def cuda_meminfo():
    if not torch.cuda.is_available():
        return

    print(
        "Total:", torch.cuda.memory_allocated() / 2 ** 30, " GB Cached: ", torch.cuda.memory_reserved() / 2 ** 30, "GB"
    )
    print(
        "Max Total:",
        torch.cuda.max_memory_allocated() / 2 ** 30,
        " GB Max Cached: ",
        torch.cuda.max_memory_reserved() / 2 ** 30,
        "GB",
    )