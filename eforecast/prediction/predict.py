import os
import joblib
import copy

import pandas as pd
import numpy as np

from joblib import Parallel
from joblib import delayed

from eforecast.datasets.data_feeder import DataFeeder
from eforecast.datasets.data_preprocessing.data_scaling import Scaler

from eforecast.clustering.clustering_manager import ClusterOrganizer
from eforecast.shallow_models.shallow_model import ShallowModel
from eforecast.combine_predictions.shallow_classifier import ShallowModelClassifier
from eforecast.common_utils.dataset_utils import sync_datasets
from eforecast.common_utils.dataset_utils import upsample_dataset
from eforecast.common_utils.dataset_utils import upsample_tensor
from eforecast.common_utils.dataset_utils import sync_target_with_tensors

try:
    import torch
    try:
        from eforecast.deep_models.pytorch_2x.network import DeepNetwork
    except:
        raise ImportError('Cannot import DeepNetwork libraries. ERRORS in network.py or global_network.py')
except:
    import tensorflow as tf
    from eforecast.deep_models.tf_2x.network import DeepNetwork

from eforecast.combine_predictions.algorithms import kmeans_predict
from eforecast.combine_predictions.algorithms import shallow_classifier_weighted_sum

CategoricalFeatures = ['hour', 'month', 'sp_index', 'dayweek']


class Predictor:
    def __init__(self, static_data, online=False, train=False):
        self.n_jobs = None
        self.static_data = static_data
        self.online = online
        self.train = train

        self.is_Fuzzy = self.static_data['is_Fuzzy']
        self.n_jobs = self.static_data['n_jobs']
        self.scale_target_method = self.static_data['scale_target_method']

        self.predictions = dict()

        self.cluster_gpu_methods = [method for method, values in static_data['cluster_methods'].items() if values]
        self.cluster_cpu_methods = [method for method, values in static_data['methods_cpu'].items() if values]
        self.methods = self.cluster_gpu_methods + self.cluster_cpu_methods

        self.global_methods = [method for method, values in static_data['global_methods'].items() if values]

        self.combine_methods = self.static_data['combining']['methods']
        self.regressors = []
        if self.is_Fuzzy:
            self.cluster_dates = dict()
            self.clusterer = ClusterOrganizer(static_data, is_online=online, train=train)
            self.predictions['clusterer'] = dict()
            self.clusters = joblib.load(os.path.join(static_data['path_model'], 'clusters.pickle'))
            self.predictions['clusters'] = dict()
            for m in self.clusterer.methods:
                if m in self.static_data['clustering']['prediction_for_method'] \
                        and self.static_data['horizon_type'] != 'multi-output':
                    self.regressors.append({'method': m})
                if m == self.clusterer.make_clusters_for_method or self.clusterer.make_clusters_for_method == 'both':
                    self.predictions['clusters'][m] = dict()
                    _, self.cluster_dates[m] = self.clusterer.compute_activations(m)
            for cluster_name, cluster_path in self.clusters.items():
                clusterer_method = [m for m in self.clusterer.methods if m in cluster_name]
                if len(clusterer_method) == 0:
                    raise ValueError(f'Cannot correspond the cluster {cluster_name} with a known clusterer method')
                clusterer_method = clusterer_method[0]
                self.predictions['clusters'][clusterer_method][cluster_name] = dict()
                for method in self.methods:
                    if not os.path.exists(os.path.join(cluster_path, method)):
                        continue
                    for path in os.listdir(os.path.join(cluster_path, method)):
                        if 'test' in path:
                            file_regressor = os.path.join(cluster_path, method, path, 'net_weights.pickle')
                            if not os.path.exists(file_regressor):
                                continue
                            self.regressors.append({'file': file_regressor,
                                                    'method': f'{method}_{path}',
                                                    'clusterer_method': clusterer_method,
                                                    'cluster_name': cluster_name,
                                                    'cluster_path': cluster_path})
        if len(self.global_methods) > 0:
            self.predictions['global'] = dict()
        for method in self.global_methods:
            cluster_name = 'global'
            cluster_path = os.path.join(static_data['path_model'], 'global')
            if not os.path.exists(os.path.join(cluster_path, method)):
                continue
            for path in os.listdir(os.path.join(cluster_path, method)):
                if 'test' in path:
                    file_regressor = os.path.join(cluster_path, method, path, 'net_weights.pickle')
                    if not os.path.exists(file_regressor):
                        continue
                    self.regressors.append({'file': file_regressor,
                                            'method': f'{method}_{path}',
                                            'cluster_name': cluster_name,
                                            'cluster_path': cluster_path})
        self.horizon = self.static_data['target_variable']['lags']
        self.data_feeder = DataFeeder(static_data, online=self.online, train=self.train)


    def init_method(self, regressor):
        if regressor['method'].split('_')[0] in self.cluster_gpu_methods + self.global_methods:
            print(os.path.dirname(regressor['file']))
            return DeepNetwork(self.static_data, os.path.dirname(regressor['file']), train=self.train,
                               online=self.online)

        elif regressor['method'].split('_')[0] in self.cluster_cpu_methods:
            print(os.path.dirname(regressor['file']))
            return ShallowModel(self.static_data, os.path.dirname(regressor['file']), train=self.train,
                               online=self.online)
        elif regressor['method'] in self.clusterer.methods:
            return regressor['method']
        else:
            raise ValueError(f"Unknown method for prediction {regressor['method']}")

    def predict_cluster(self, regressor):
        model = self.init_method(regressor)

        if 'cluster_dir' in model.params.keys():
            model.params['cluster_dir'] = regressor['cluster_path']

        if 'cluster' in model.params.keys():
            model.params['cluster']['cluster_path'] = regressor['cluster_path']

        if regressor['cluster_name'] != 'global':
            name = '_'.join(regressor['cluster_name'].split('_')[1:])
            cluster_dates = self.cluster_dates[regressor['clusterer_method']][name]
            if cluster_dates.shape[0] == 0:
                return None
        else:
            cluster_dates = None

        return model.predict(cluster_dates=cluster_dates).clip(0, np.inf)


    def predict_clusterer(self):
        pred, _ = self.clusterer.predict('RBF')
        return pred.clip(0, np.inf)

    def predict_func(self, regressor):
        if regressor['method'] == 'RBF':
            pred = self.predict_clusterer()
        elif 'clusterer_method' in regressor.keys() or regressor['cluster_name'] == 'global':
            pred = self.predict_cluster(regressor)
        else:
            pred = None
        return regressor, pred

    def predict_single_regressor(self, clusterer_method, cluster_name):
        preds = []
        for regressor in self.regressors:
            if 'clusterer_method' in regressor.keys():
                if clusterer_method == regressor['clusterer_method'] and cluster_name == regressor['cluster_name']:
                    preds.append(self.predict_func(regressor))
        for pred_reg in preds:
            regressor, pred = pred_reg
            method = regressor['method']
            self.predictions['clusters'][clusterer_method][cluster_name][method] = pred
        self.save_predictions()

    def sum_dfs(self, dfs):
        df = None
        for df1 in dfs:
            if df is None:
                df = df1
            else:
                df += df1
        return df / len(dfs)

    def predict_regressors(self, average=False, parallel=False):
        if parallel:
            preds = Parallel(n_jobs=8)(delayed(self.predict_func)(regressor) for regressor in self.regressors)
        else:
            preds = [self.predict_func(regressor) for regressor in self.regressors]

        for pred_reg in preds:
            regressor, pred = pred_reg
            if pred is not None:
                if 'clusterer_method' in regressor.keys():
                    clusterer_method = regressor['clusterer_method']
                    cluster_name = regressor['cluster_name']
                    method = regressor['method']
                    self.predictions['clusters'][clusterer_method][cluster_name][method] = pred
                elif regressor['method'] == 'RBF':
                    self.predictions['clusterer'][regressor['method']] = pred
                elif regressor['cluster_name'] == 'global':
                    self.predictions['global'][regressor['method']] = pred
        if average:
            if 'clusters' in self.predictions.keys():
                for clusterer_method, rules in self.predictions['clusters'].items():
                    for cluster_name, methods_predictions in rules.items():
                        methods = np.unique([m.split('_')[0] for m in methods_predictions.keys()])
                        pred = dict()
                        for method in methods:
                            dfs =[df for n, df in methods_predictions.items() if method in n]
                            pred[method] = self.sum_dfs(dfs)
                        self.predictions['clusters'][clusterer_method][cluster_name] = pred
            if len(self.global_methods) > 0:
                methods = np.unique([m.split('_')[0] for m in self.predictions['global'].keys()])
                pred = dict()
                for method in methods:
                    dfs = [df for n, df in self.predictions['global'].items() if method in n]
                    pred[method] = self.sum_dfs(dfs)
                self.predictions['global'] = pred
        self.save_predictions()

    def save_predictions(self):
        if not self.online:
            if self.train:
                joblib.dump(self.predictions, os.path.join(self.static_data['path_data'],
                                                           'predictions_regressors_train.pickle'))
            else:
                joblib.dump(self.predictions, os.path.join(self.static_data['path_data'],
                                                           'predictions_regressors_eval.pickle'))
        else:
            joblib.dump(self.predictions, os.path.join(self.static_data['path_data'],
                                                       'predictions_regressors_online.pickle'))

    def load_predictions(self):
        if not self.online:
            if self.train:
                self.predictions = joblib.load(os.path.join(self.static_data['path_data'],
                                                            'predictions_regressors_train.pickle'))
            else:
                self.predictions = joblib.load(os.path.join(self.static_data['path_data'],
                                                            'predictions_regressors_eval.pickle'))
        else:
            self.predictions = joblib.load(os.path.join(self.static_data['path_data'],
                                                        'predictions_regressors_online.pickle'))

    def inverse_transform_predictions(self, pred):
        scaler = Scaler(self.static_data, recreate=False, online=self.online, train=self.train)
        return scaler.inverse_transform_data(pred, f'target_{self.scale_target_method}')

    def predict_combine_algorithms(self, combine_method, pred_methods, X_inputs, hor, cluster_name, path_combine_method,
                                   cluster_dir):
        if combine_method in self.methods:
            combine_method = f'{combine_method}_classifier'
        cols = [f'{combine_method}_{col}' for col in self.static_data['target_variable']['columns']][hor]
        if pred_methods.shape[0] == 0:
            return pd.DataFrame(columns=cols)
        pred = None
        if 'CatBoost' not in combine_method:
            model = joblib.load(os.path.join(path_combine_method,
                                             f'{combine_method}_model.pickle'))
            if combine_method == 'bcp':
                pred = np.matmul(model['w'], pred_methods.values.T).T / np.sum(model['w'])
            elif combine_method == 'kmeans':
                pred = kmeans_predict(model, pred_methods, X_inputs, self.n_jobs)
            elif combine_method == 'elastic_net':
                pred = model.predict(pred_methods.values)
            else:
                ValueError(f'Unknown combine method {combine_method}')
        else:
            model = ShallowModelClassifier(self.static_data, path_combine_method, train=self.train, online=self.online)
            params = model.params
            predictors_id = model.predictors
            proba = model.predict_proba(cluster_dates=pred_methods.index)
            if predictors_id is None:
                predictors_id = np.arange(len(pred_methods.columns))
            pred = shallow_classifier_weighted_sum(proba, pred_methods.iloc[:, predictors_id], self.n_jobs)

        pred = pred.clip(0, np.inf)
        return pd.DataFrame(pred, index=pred_methods.index, columns=[cols])

    def predict_combine_methods_for_cluster(self, clusterer_method, cluster_name, trial=None):
        methods_predictions = self.predictions['clusters'][clusterer_method][cluster_name]
        if cluster_name == 'averages':
            return methods_predictions
        scale_nwp_method = self.static_data['combining']['data_type']['scale_nwp_method']
        scale_row_method = self.static_data['combining']['data_type']['scale_row_method']
        merge = self.static_data['combining']['data_type']['merge']
        compress = self.static_data['combining']['data_type']['compress']
        what_data = 'row_all'
        x, mdata = self.data_feeder.feed_inputs(what_data, merge=merge, compress=compress,
                                                scale_nwp_method=scale_nwp_method,
                                                scale_row_method=scale_row_method,
                                                feature_selection_method=None,
                                                cluster={'cluster_name': 'global',
                                                         'cluster_path': os.path.join(self.static_data['path_model'],
                                                                                      'global')})
        X_inputs = x[what_data]
        metadata = mdata[what_data]

        for combine_method in self.combine_methods:
            if combine_method in self.methods:
                combine_method = f'{combine_method}_classifier'
            if len(self.methods) > 1:
                methods_predictions[combine_method] = pd.DataFrame()
        for hor in self.horizon:
            path_combine_cluster = os.path.join(self.clusters[cluster_name], 'combine')
            n_predictors = len(self.methods)
            if n_predictors > 1:
                pred_methods = []
                for method in sorted(self.methods):
                    pred1 = methods_predictions[method].iloc[:, hor].to_frame()
                    pred1.columns = [method]
                    pred_methods.append(pred1)
                pred_methods = pd.concat(pred_methods, axis=1)
                if pred_methods.shape[0] > 0:
                    pred_methods[pred_methods < 0] = 0
                    pred_methods = pred_methods.dropna(axis='index')

                for combine_method in self.combine_methods:
                    path_combine_method = os.path.join(path_combine_cluster, combine_method)
                    if trial is not None:
                        path_combine_method = os.path.join(path_combine_method, f'trial_{trial}')
                    if self.static_data['horizon_type'] == 'multi-output':
                        path_combine_method = os.path.join(path_combine_method, f'hour_ahead_{hor}')
                    if not os.path.exists(path_combine_method):
                        raise ImportError(f'Cannot find weights for combine method {combine_method} of'
                                          f' cluster {cluster_name}\n'
                                          f' at the folder {path_combine_method}')

                    pred = self.predict_combine_algorithms(combine_method, pred_methods, X_inputs, hor,
                                                           cluster_name, path_combine_method,
                                                           self.clusters[cluster_name])
                    if combine_method in self.methods:
                        combine_method = f'{combine_method}_classifier'
                    if pred.shape[0] > 0:
                        pred = pred.clip(0, np.inf)
                    methods_predictions[combine_method] = pd.concat([methods_predictions[combine_method],
                                                                     pred],
                                                                    axis=1)
        self.predictions['clusters'][clusterer_method][cluster_name] = methods_predictions
        self.save_predictions()

    def predict_combine_methods(self):
        self.load_predictions()
        scale_nwp_method = self.static_data['combining']['data_type']['scale_nwp_method']
        scale_row_method = self.static_data['combining']['data_type']['scale_row_method']
        merge = self.static_data['combining']['data_type']['merge']
        compress = self.static_data['combining']['data_type']['compress']
        what_data = 'row_all'
        x, mdata = self.data_feeder.feed_inputs(what_data, merge=merge, compress=compress,
                                                scale_nwp_method=scale_nwp_method,
                                                scale_row_method=scale_row_method,
                                                feature_selection_method=None,
                                                cluster={'cluster_name': 'global',
                                                         'cluster_path': os.path.join(self.static_data['path_model'],
                                                                                      'global')})
        X_inputs = x[what_data]
        metadata = mdata[what_data]
        if self.is_Fuzzy:
            for clusterer_method, rules in self.predictions['clusters'].items():
                for cluster_name, methods_predictions in rules.items():
                    if cluster_name == 'averages':
                        continue
                    for combine_method in self.combine_methods:
                        if combine_method in self.methods:
                            combine_method = f'{combine_method}_classifier'
                        if len(self.methods) > 1:
                            methods_predictions[combine_method] = pd.DataFrame()
                    for hor in self.horizon:
                        path_combine_cluster = os.path.join(self.clusters[cluster_name], 'combine')
                        n_predictors = len(self.methods)
                        if n_predictors > 1:
                            pred_methods = []
                            for method in sorted(self.methods):
                                pred1 = methods_predictions[method].iloc[:, hor].to_frame()
                                pred1.columns = [method]
                                pred_methods.append(pred1)
                            pred_methods = pd.concat(pred_methods, axis=1)
                            if pred_methods.shape[0] > 0:
                                pred_methods[pred_methods < 0] = 0
                                pred_methods = pred_methods.dropna(axis='index')

                            for combine_method in self.combine_methods:
                                path_combine_method = os.path.join(path_combine_cluster, combine_method)
                                if self.static_data['horizon_type'] == 'multi-output':
                                    path_combine_method = os.path.join(path_combine_method, f'hour_ahead_{hor}')
                                if not os.path.exists(path_combine_method):
                                    raise ImportError(f'Cannot find weights for combine method {combine_method} of'
                                                      f' cluster {cluster_name}\n'
                                                      f' at the folder {path_combine_method}')

                                pred = self.predict_combine_algorithms(combine_method, pred_methods, X_inputs, hor,
                                                                       cluster_name, path_combine_method,
                                                                       self.clusters[cluster_name])
                                if combine_method in self.methods:
                                    combine_method = f'{combine_method}_classifier'
                                if pred.shape[0] > 0:
                                    pred = pred.clip(0, np.inf)
                                methods_predictions[combine_method] = pd.concat([methods_predictions[combine_method],
                                                                                 pred],
                                                                                axis=1)
            self.save_predictions()

    def compute_predictions_averages(self, only_methods=False, only_combine_methods=False):
        self.load_predictions()
        scale_nwp_method = self.static_data['combining']['data_type']['scale_nwp_method']
        scale_row_method = self.static_data['combining']['data_type']['scale_row_method']
        merge = self.static_data['combining']['data_type']['merge']
        compress = self.static_data['combining']['data_type']['compress']
        what_data = 'row_all'
        cluster_name = 'global'
        cluster_path = os.path.join(self.static_data['path_model'], 'global')
        x, mdata = self.data_feeder.feed_inputs(what_data, merge=merge, compress=compress,
                                                scale_nwp_method=scale_nwp_method,
                                                scale_row_method=scale_row_method,
                                                feature_selection_method=None,
                                                cluster={'cluster_name': cluster_name,
                                                         'cluster_path': cluster_path})
        X_inputs = x[what_data]
        metadata = mdata[what_data]
        methods = [m for m in self.methods] if not only_combine_methods else []
        if not only_methods:
            for cm in self.combine_methods:
                methods.append(f'{cm}_classifier' if cm in self.methods else cm)
        if self.is_Fuzzy:
            for clusterer_method, rules in self.predictions['clusters'].items():
                if 'averages' not in self.predictions['clusters'][clusterer_method].keys():
                    self.predictions['clusters'][clusterer_method]['averages'] = dict()
                for method in methods:
                    method_predictions = pd.DataFrame(index=metadata['dates'])
                    for hor in self.horizon:
                        horizon_predictions = pd.DataFrame(index=metadata['dates'])
                        col = [f'{method}_{col}' for col in self.static_data['target_variable']['columns']][hor]
                        for cluster_name, methods_predictions in rules.items():
                            if cluster_name in self.clusters.keys() and \
                                    method in self.predictions['clusters'][clusterer_method][cluster_name]:
                                pred = self.predictions['clusters'][clusterer_method][cluster_name][method][col] \
                                    .to_frame(f'{cluster_name}_{col}')
                                horizon_predictions = horizon_predictions.join(pred)
                        if horizon_predictions.shape[1] > 0:
                            horizon_predictions = horizon_predictions.clip(0, np.inf)
                            horizon_predictions = horizon_predictions.mean(axis=1).to_frame(f'{col}_average')
                            horizon_predictions = horizon_predictions.dropna(axis='index')
                            method_predictions = pd.concat([method_predictions, horizon_predictions], axis=1)
                    if method_predictions.shape[1] > 0:
                        self.predictions['clusters'][clusterer_method]['averages'][
                            f'{method}_average'] = method_predictions

            self.save_predictions()

    def predict_combine_models(self, combine_methods=None):
        self.load_predictions()
        self.predictions['models'] = dict()
        if combine_methods is None:
            print('No ML combine methods provided')
            self.save_predictions()
            return
        scale_nwp_method = self.static_data['combining']['data_type']['scale_nwp_method']
        scale_row_method = self.static_data['combining']['data_type']['scale_row_method']
        merge = self.static_data['combining']['data_type']['merge']
        compress = self.static_data['combining']['data_type']['compress']
        what_data = 'row_all'
        cluster_name = 'global'
        cluster_path = os.path.join(self.static_data['path_model'], 'global')
        x, mdata = self.data_feeder.feed_inputs(what_data, merge=merge, compress=compress,
                                                scale_nwp_method=scale_nwp_method,
                                                scale_row_method=scale_row_method,
                                                feature_selection_method=None,
                                                cluster={'cluster_name': cluster_name,
                                                         'cluster_path': cluster_path})
        X_inputs = x[what_data]
        metadata = mdata[what_data]
        alias_methods = []
        for cm in self.combine_methods:
            if cm in self.methods:
                cm = f'{cm}_classifier'
            alias_methods.append(cm)
        for cm in combine_methods:
            if cm in self.methods:
                cm = f'{cm}_classifier'
            self.predictions['models'][cm] = pd.DataFrame()

        for hor in self.horizon:
            pred_models = []
            if 'global' in self.predictions.keys():
                for global_model, global_prediction in self.predictions['global'].items():
                    pred_models.append(global_prediction.iloc[:, hor].to_frame())
            if 'clusters' in self.predictions.keys():
                for clusterer_method, rules in self.predictions['clusters'].items():
                    for combine_method, combine_prediction in rules['averages'].items():
                        pred_models.append(combine_prediction.iloc[:, hor].
                                           to_frame(f'{clusterer_method}_{combine_prediction.columns[hor]}'))
            n_predictors = len(pred_models)
            pred_models = pd.concat(pred_models, axis=1)
            pred_models = pred_models.clip(0, np.inf)
            pred_models = pred_models.dropna(axis='index')
            if n_predictors > 1 and combine_methods is not None:
                for combine_method in combine_methods:
                    print(f'Make predictions for combine method {combine_method} for models and horizon {hor}')
                    path_combine_method = os.path.join(self.static_data['path_model'], 'combine_models',
                                                       combine_method)
                    if self.static_data['horizon_type'] == 'multi-output':
                        path_combine_method = os.path.join(path_combine_method, f'hour_ahead_{hor}')
                    if not os.path.exists(path_combine_method):
                        raise ImportError(f'Cannot find weights for combine method {combine_method} of'
                                          f' models\n'
                                          f' at the folder {path_combine_method}')
                    pred = self.predict_combine_algorithms(combine_method, pred_models, X_inputs, hor,
                                                           cluster_name, path_combine_method,
                                                           cluster_path)
                    if combine_method in self.methods:
                        combine_method = f'{combine_method}_classifier'
                    pred = pred.clip(0, np.inf)
                    self.predictions['models'][combine_method] = pd.concat([self.predictions['models'][combine_method],
                                                                            pred], axis=1)
            elif n_predictors == 1:
                self.predictions['models'][combine_methods[0]] = pd.concat([self.predictions['models'][combine_methods[0]],
                                                                            pred_models], axis=1)
            self.save_predictions()

