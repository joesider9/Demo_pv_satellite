import copy
import os
import joblib
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from eforecast.clustering.clustering_manager import ClusterOrganizer
from eforecast.datasets.files_manager import FilesManager

from eforecast.common_utils.dataset_utils import sync_datasets
from eforecast.datasets.data_feeder import DataFeeder
from eforecast.datasets.data_preprocessing.data_scaling import Scaler

from eforecast.combine_predictions.algorithms import bcp_fit
from eforecast.combine_predictions.algorithms import kmeans_fit
from eforecast.combine_predictions.train_combine_classifier import train_classifier
from sklearn.linear_model import ElasticNetCV

CategoricalFeatures = ['hour', 'month', 'sp_index', 'dayweek']

class CombinerFit:
    def __init__(self, static_data, refit=False):
        self.kmeans = None
        self.y_resample = None
        self.num_feats = []
        self.labels = None
        self.cat_feats = []
        self.metadata = None
        self.X = None
        self.y = None
        self.static_data = static_data
        self.refit = refit
        self.is_Fuzzy = self.static_data['is_Fuzzy']
        if self.is_Fuzzy:
            self.clusters = joblib.load(os.path.join(static_data['path_model'], 'clusters.pickle'))
        else:
            self.clusters = dict()
        self.scale_row_method = self.static_data['combining']['data_type']['scale_row_method']
        self.scale_nwp_method = self.static_data['combining']['data_type']['scale_nwp_method']
        self.merge = self.static_data['combining']['data_type']['merge']
        self.compress = self.static_data['combining']['data_type']['compress']
        self.problem_type = self.static_data['type']
        self.n_jobs = self.static_data['n_jobs']

        self.predictions = joblib.load(os.path.join(self.static_data['path_data'],
                                                    'predictions_regressors_train.pickle'))
        self.scaler = Scaler(static_data, recreate=False, online=False, train=True)
        self.scale_target_method = self.static_data['scale_target_method']
        self.rated = self.static_data['rated']
        self.combine_methods = self.static_data['combining']['methods']
        self.cluster_gpu_methods = [method for method, values in static_data['cluster_methods'].items() if values]
        self.cluster_cpu_methods = [method for method, values in static_data['methods_cpu'].items() if values]
        self.methods = self.cluster_gpu_methods + self.cluster_cpu_methods
        self.horizon = self.static_data['target_variable']['lags']
        self.combine_clusters = dict()
        for cluster_name, cluster_dir in self.clusters.items():
            path_combine_cluster = os.path.join(cluster_dir, 'combine')
            self.combine_clusters.update({cluster_name: path_combine_cluster})
        self.data_feeder = DataFeeder(static_data, online=False, train=True)


    def feed_data(self, methods=True):
        which = 'methods' if methods else 'models'
        print(f'Read data for Combine {which}....')
        what_data = 'row_all'
        x, mdata = self.data_feeder.feed_inputs(what_data, merge=self.merge, compress=self.compress,
                                                scale_nwp_method=self.scale_nwp_method,
                                                scale_row_method=self.scale_row_method,
                                                feature_selection_method=None,
                                                cluster={'cluster_name': 'global',
                                                         'cluster_path': os.path.join(self.static_data['path_model'],
                                                                                      'global')})
        self.X = x[what_data]
        self.metadata = mdata[what_data]

        self.y = self.data_feeder.feed_target(inverse=False)
        self.X, self.y = sync_datasets(self.X, self.y)

        self.metadata['dates'] = self.X.index
        self.cat_feats = [v_name for v_name in self.X.columns
                          for c_feats in CategoricalFeatures if c_feats in v_name]
        self.cat_feats = list(set([v_name for v_name in self.X.columns
                                   if len(np.unique(self.X[v_name].values)) < 30]))

    def apply_kmeans_X(self):
        n_clusters = 5 if self.problem_type == 'pv' else 12
        cat_feats = [f for f in self.cat_feats if 'hour' not in f]
        if len(cat_feats):
            self.kmeans = KMeans(n_clusters=n_clusters)
            self.labels = pd.Series(self.kmeans.fit_predict(self.X[cat_feats].values), index=self.X.index,
                                    name='labels')

    def fit_combine_method(self, combine_method, pred_methods, y, hor, n_predictors, dates, cluster_name,
                           path_combine_method, cluster_dir):
        if not os.path.exists(os.path.join(path_combine_method, f'{combine_method}_model.pickle')) \
                or self.refit:
            if combine_method == 'bcp':
                print('BCP training')
                model = dict()
                w = bcp_fit(pred_methods.values, y.iloc[:, hor].values.reshape(-1, 1),
                            n_predictors)
                model['w'] = w
            elif combine_method == 'kmeans':
                print('Kmeans training')
                n_clusters = 5 if self.problem_type == 'pv' else 12
                if dates.shape[0] < 50:
                    n_clusters = 2
                kmeans = KMeans(n_clusters=n_clusters)
                labels = pd.Series(kmeans.fit_predict(self.X.loc[dates].values),
                                   index=dates,
                                   name='labels')
                kmeans_model = kmeans_fit(kmeans, labels.values, pred_methods.values,
                                          y.iloc[:, hor].values.reshape(-1, 1))
                model = kmeans_model

            elif combine_method == 'elastic_net':
                print('elastic_net training')
                model = ElasticNetCV(cv=5, max_iter=200000)
                model.fit(pred_methods.values, y.iloc[:, hor].values)
            else:
                if not os.path.exists(os.path.join(path_combine_method, f'results_{cluster_name}_{combine_method}.csv')) \
                        or self.refit:
                    best_predictor = np.argmin(np.abs(pred_methods.values -
                                                      y.iloc[:, hor].values.reshape(-1, 1)), axis=1).reshape(-1, 1)
                    classes = np.unique(best_predictor)
                    predictors_id = []
                    for cl in classes:
                        count = np.where(best_predictor == cl)[0].shape[0]
                        if count > 8:
                            predictors_id.append(cl)
                    best_predictor = np.argmin(np.abs(pred_methods.values[:, predictors_id] -
                                                      y.iloc[:, hor].values.reshape(-1, 1)), axis=1).reshape(-1, 1)
                    best_predictor = pd.DataFrame(best_predictor, index=dates, columns=['target'])
                    train_classifier(self.static_data, combine_method, cluster_name,
                                     path_combine_method, cluster_dir,
                                     best_predictor, predictors_id, refit=self.refit)
                    model = None
                else:
                    model = None
            if model is not None:
                joblib.dump(model, os.path.join(path_combine_method,
                                                f'{combine_method}_model.pickle'))

    def fit_methods(self):
        self.feed_data()
        predictions = self.predictions
        if self.is_Fuzzy:
            for hor in self.horizon:
                for clusterer_method, rules in predictions['clusters'].items():
                    for cluster_name, methods_predictions in rules.items():
                        if cluster_name == 'averages':
                            continue
                        n_predictors = len(methods_predictions)
                        if n_predictors > 1:
                            cv_masks = joblib.load(os.path.join(self.clusters[cluster_name], 'cv_mask.pickle'))
                            cv_mask = cv_masks[1].union(cv_masks[-1])
                            pred_methods = []
                            for method in sorted(methods_predictions.keys()):
                                pred = methods_predictions[method].iloc[:, hor].to_frame()
                                pred.columns = [method]
                                pred_methods.append(pred)
                            pred_methods = pd.concat(pred_methods, axis=1)
                            pred_methods[pred_methods < 0] = 0
                            pred_methods = pred_methods.dropna(axis='index')
                            dates = pred_methods.index.intersection(self.y.index)
                            dates = dates.intersection(cv_mask)
                            pred_methods = pred_methods.loc[dates]
                            y = self.y.loc[dates]
                            for combine_method in self.combine_methods:
                                print(f'Fitting combine method {combine_method} for cluster {cluster_name} '
                                      f'and horizon {hor}')
                                path_combine_method = os.path.join(self.combine_clusters[cluster_name],
                                                                   combine_method)
                                if self.static_data['horizon_type'] == 'multi-output':
                                    path_combine_method = os.path.join(path_combine_method, f'hour_ahead_{hor}')
                                if not os.path.exists(path_combine_method):
                                    os.makedirs(path_combine_method)
                                self.fit_combine_method(combine_method, pred_methods, y, hor,
                                                        n_predictors, dates, cluster_name,
                                                        path_combine_method, self.clusters[cluster_name])

    def fit_methods_for_cluster(self, clusterer_method, cluster_name, trial=None):
        methods_predictions = self.predictions['clusters'][clusterer_method][cluster_name]
        self.feed_data()
        cv_masks = joblib.load(os.path.join(self.clusters[cluster_name], 'cv_mask.pickle'))
        cv_mask = cv_masks[1].union(cv_masks[-1])
        for hor in self.horizon:
            n_predictors = len(methods_predictions)
            if n_predictors > 1:
                pred_methods = []
                for method in sorted(methods_predictions.keys()):
                    pred = methods_predictions[method].iloc[:, hor].to_frame()
                    pred.columns = [method]
                    pred_methods.append(pred)
                pred_methods = pd.concat(pred_methods, axis=1)
                pred_methods[pred_methods < 0] = 0
                pred_methods = pred_methods.dropna(axis='index')
                dates = pred_methods.index.intersection(self.y.index)
                dates = dates.intersection(cv_mask)
                pred_methods = pred_methods.loc[dates]
                y = self.y.loc[dates]
                for combine_method in self.combine_methods:
                    print(f'Fitting combine method {combine_method} for cluster {cluster_name} '
                          f'and horizon {hor}')
                    path_combine_method = os.path.join(self.combine_clusters[cluster_name],
                                                       combine_method)
                    if trial is not None:
                        path_combine_method = os.path.join(path_combine_method, f'trial_{trial}')
                    if self.static_data['horizon_type'] == 'multi-output':
                        path_combine_method = os.path.join(path_combine_method, f'hour_ahead_{hor}')
                    if not os.path.exists(path_combine_method):
                        os.makedirs(path_combine_method)
                    self.fit_combine_method(combine_method, pred_methods, y, hor,
                                            n_predictors, dates, cluster_name,
                                            path_combine_method, self.clusters[cluster_name])

    def fit_models(self, combine_methods):
        cluster_name = 'global'
        cluster_path = os.path.join(self.static_data['path_model'], 'global')
        alias_methods = []
        for cm in self.combine_methods:
            alias_methods.append(f'{cm}_classifier' if cm in self.methods else cm)
        self.feed_data(methods=False)
        for hor in self.horizon:
            pred_models = []
            if 'global' in self.predictions.keys():
                for global_model, global_prediction in self.predictions['global'].items():
                    pred_models.append(global_prediction.iloc[:, hor].to_frame())
            if 'clusters' in self.predictions.keys():
                for clusterer_method, rules in self.predictions['clusters'].items():
                    for combine_method, combine_prediction in rules['averages'].items():
                        if self.static_data['horizon_type'] == 'multi-output':
                            pred_models.append(combine_prediction.iloc[:, hor].
                                               to_frame(f"{clusterer_method}_"
                                                        f"{'_'.join(combine_method.split('_')[:-1])}"
                                                        f"_{combine_prediction.columns[hor]}"))
                        else:
                            pred_models.append(combine_prediction.iloc[:, hor].
                                               to_frame(f'{clusterer_method}_{combine_prediction.columns[hor]}'))
            n_predictors = len(pred_models)
            pred_models = pd.concat(pred_models, axis=1)
            pred_models = pred_models.clip(0, np.inf)
            pred_models = pred_models.dropna(axis='index')
            dates = pred_models.index.intersection(self.y.index)
            pred_models = pred_models.loc[dates]
            y = self.y.loc[dates]
            if n_predictors > 1:
                for combine_method in combine_methods:
                    print(f'Fitting combine method {combine_method} for models and horizon {hor}')
                    path_combine_method = os.path.join(self.static_data['path_model'], 'combine_models',
                                                       combine_method)
                    if self.static_data['horizon_type'] == 'multi-output':
                        path_combine_method = os.path.join(path_combine_method, f'hour_ahead_{hor}')
                    if not os.path.exists(path_combine_method):
                        os.makedirs(path_combine_method)
                    self.fit_combine_method(combine_method, pred_models, y, hor,
                                            n_predictors, dates, cluster_name,
                                            path_combine_method, cluster_path)
