import traceback
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed

from eforecast.training.threading.train_manager_for_gpu import train_on_gpus
from eforecast.training.threading.train_transfer_learning_for_gpu import train_transfer_learning_on_gpus
from eforecast.training.train_rbfnns_on_cpus import train_rbfnn_on_cpus
from eforecast.training.train_clustrers_on_cpu import train_clusters_on_cpus


def fit_on_gpus(static_data, cluster=None, global_method=None,
                cluster_method=None, refit=False):
    if static_data['transfer_learning']:
        train_transfer_learning_on_gpus(static_data, static_data['static_data_base'], refit=refit)
    else:
        train_on_gpus(static_data, cluster=cluster, global_method=global_method,
                  cluster_method=cluster_method, refit=refit)
    return 'Done'


def fit_on_cpus(static_data, cluster=None, method=None, refit=False):
    if static_data['is_Fuzzy']:
        train_rbfnn_on_cpus(static_data, cluster=cluster, method=method, refit=refit)
        train_clusters_on_cpus(static_data, cluster=cluster, method=method, refit=refit)
    return 'Done'


def fit_clusters(static_data, cluster=None, global_gpu_method=None,
                cluster_gpu_method=None, cluster_cpu_method=None, refit=False):
    # r= fit_on_gpus(static_data, cluster=cluster,
    #                                    global_method=global_gpu_method,
    #                                    cluster_method=cluster_gpu_method, refit=refit)
    res = []
    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(fit_on_gpus, static_data, cluster=cluster,
                                       global_method=global_gpu_method,
                                       cluster_method=cluster_gpu_method, refit=refit),
                       executor.submit(fit_on_cpus, static_data, cluster=cluster,
                                       method=cluster_cpu_method, refit=refit)]
            for future in as_completed(futures):
                res.append(future.result())
    except Exception as e:
        tb = traceback.format_exception(e)
        print("".join(tb))
        return "".join(tb)

    return 'Done'
