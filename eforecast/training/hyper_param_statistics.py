import os
import joblib

import pandas as pd
import matplotlib.pyplot as plt


def hyper_param_methods(static_data, cluster=None, method=None):
    if method is None:
        methods = [method for method, values in static_data['project_methods'].items() if
                   values]
    else:
        methods = [method]

    for m in methods:
        best_trials = pd.DataFrame()
        if cluster is not None:
            clusters = cluster
        else:
            clusters = joblib.load(os.path.join(static_data['path_model'], 'clusters.pickle'))
        for cluster_name, cluster_dir in clusters.items():
            results = pd.read_csv(os.path.join(cluster_dir, f'results_{cluster_name}_{m}.csv'), index_col=0,
                                  header=0)
            best_trials = pd.concat([best_trials, results.iloc[:4]])
        if 'feature_selection_method' in best_trials.columns:
            best_trials.feature_selection_method[best_trials.feature_selection_method.isna()] = 'full'
        path = os.path.join(static_data['path_model'], 'hyper_param_stats', m)
        if not os.path.exists(path):
            os.makedirs(path)
        for col in best_trials.columns:
            if 'mae' not in col and 'sse' not in col and col not in {'value', 'duration', 'state', 'params_objective'} \
                    and 'date' not in col:
                plt.figure()
                try:
                    best_trials[col].dropna().hist(bins=12)
                except:
                    best_trials[col].dropna().astype('str').hist(bins=12)
                plt.savefig(os.path.join(path, f'{col}.png'))
                plt.close()
