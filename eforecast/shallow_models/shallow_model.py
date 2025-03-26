import os
import joblib

import numpy as np
import pandas as pd

from threadpoolctl import threadpool_limits

from sklearn.linear_model import LassoCV, Lasso
from sklearn.linear_model import MultiTaskLassoCV, MultiTaskLasso
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor
from catboost import Pool

from eforecast.shallow_models.rbf_ols_network import RBFols
from eforecast.datasets.data_feeder import DataFeeder
from eforecast.common_utils.dataset_utils import sync_datasets
from eforecast.common_utils.dataset_utils import sync_dict_df

pd.set_option('display.expand_frame_repr', False)

CategoricalFeatures = ['dayweek', 'hour', 'month', 'sp_index']


class ShallowModel:
    def __init__(self, static_data, path_weights, params=None, n_jobs=1, refit=False, train=False, online=False):
        self.best_mae_val = None
        self.best_mae_test = None
        self.is_trained = False
        self.refit = refit
        self.static_data = static_data
        self.rated = static_data['rated']
        self.n_jobs = n_jobs
        if params is not None:
            self.params = params
            self.method = self.params['method']
            self.cluster_name = self.params['cluster_name']
            self.merge = self.params['merge']
            self.compress = self.params['compress']
            self.scale_nwp_method = self.params['scale_nwp_method']
            self.scale_row_method = self.params['scale_row_method']
            if self.method == 'RF':
                self.model = RandomForestRegressor(n_jobs=self.n_jobs)
                self.best_params = self.model.get_params()
            elif self.method == 'CatBoost':
                self.model = CatBoostRegressor(thread_count=self.n_jobs, allow_writing_files=False)
                self.best_params = {'iterations': 1000,
                                    'learning_rate': 0.005,
                                    'l2_leaf_reg': 1,
                                    "objective": "RMSE",
                                    'min_data_in_leaf': 2,
                                    "colsample_bylevel": 0.3,
                                    "depth": 1,
                                    "boosting_type": "Ordered",
                                    "bootstrap_type": "Bayesian",
                                    "eval_metric": "MAE"}
            elif self.method == 'lasso':
                if self.static_data['horizon_type'] == 'multi-output':
                    self.model = MultiTaskLassoCV(max_iter=150000, n_jobs=self.n_jobs)
                else:
                    self.model = LassoCV(max_iter=150000, n_jobs=self.n_jobs)
                self.best_params = self.model.get_params()
            elif 'RBF' in self.method:
                self.model = RBFols(self.rated)
                self.best_params = self.model.get_params()
            else:
                raise ValueError(f'Unknown method {self.method} for shallow models')
            for param, value in self.params.items():
                if param in self.best_params.keys():
                    self.best_params[param] = value
            self.model.set_params(**self.best_params)
        self.path_weights = path_weights
        try:
            if not self.refit:
                self.load()
        except:
            pass
        self.refit = refit
        self.is_online = online
        self.train = train
        self.data_feeder = DataFeeder(static_data, online=self.is_online, train=self.train)
        if not hasattr(self, 'params'):
            raise RuntimeError('The network has no params. You should train the net providing params')

    def load_data(self, with_target=False):
        what_data = self.params['what_data']
        x, mdata = self.data_feeder.feed_inputs(what_data, merge=self.params['merge'], compress=self.params['compress'],
                                                scale_nwp_method=self.params['scale_nwp_method'],
                                                scale_row_method=self.params['scale_row_method'],
                                                feature_selection_method=self.params['feature_selection_method'],
                                                cluster={'cluster_name': self.params['cluster_name'],
                                                         'cluster_path': self.params['cluster_dir']})
        X = x[what_data]
        metadata = mdata[what_data]
        if not with_target:
            return X, metadata
        else:
            y = self.data_feeder.feed_target(inverse=False)
            if isinstance(X, pd.DataFrame):
                X, y = sync_datasets(X, y)
            else:
                X, y = sync_dict_df(X, y)
            metadata['dates'] = y.index
            return X, y, metadata

    def get_rated(self, y):
        if self.static_data['rated'] is not None:
            norm_val = 1
        else:
            norm_val = y
        return norm_val

    @staticmethod
    def get_slice(x, mask, metadata, y=None):
        dates = metadata['dates']
        mask = mask.intersection(dates)
        indices = dates.get_indexer(mask)
        y_slice = y.iloc[indices] if y is not None else None
        if isinstance(x, pd.DataFrame):
            X_slice = x.iloc[indices]
        else:
            raise ValueError('Wrong type of input X for shallow models')
        return X_slice, y_slice

    def fit(self, cv_masks):
        X, y, metadata = self.load_data(with_target=True)
        if isinstance(X, dict):
            for key in sorted(X.keys()):
                X[key].columns = [f'{key}_{col}' for col in X[key].columns]
            X_ = pd.concat([X[key] for key in sorted(X.keys())], axis=1)
            X = X_.copy()

        self.cat_feats = list(set([v_name for v_name in X.columns if len(np.unique(X[v_name].values)) < 30]))

        X[self.cat_feats] = X[self.cat_feats].astype('int')
        X_train, y_train = self.get_slice(X, cv_masks[0], metadata, y=y)
        X_val, y_val = self.get_slice(X, cv_masks[1], metadata, y=y)
        X_test, y_test = self.get_slice(X, cv_masks[2], metadata, y=y)

        if self.method in {'lasso', 'RF'}:
            X_train = pd.concat([X_train, X_val]).values
            y_train = pd.concat([y_train, y_val]).values
            if y_train.shape[1] == 1:
                y_train = y_train.ravel()
            self.model.fit(X_train, y_train)
            y_pred = self.model.predict(X_test)
        elif 'RBF' in self.method:
            with threadpool_limits(limits=1):
                if y_train.shape[0] > 1200:
                    ind = np.random.randint(low=0, high=X_train.shape[0], size=1200)
                    X_train = X_train.iloc[ind]
                    y_train = y_train.iloc[ind]
                self.model.fit(X_train.values, y_train.values, X_val.values, y_val.values)
                y_pred = self.model.predict(X_test)
        else:
            try:
                self.model.fit(X_train, y_train, cat_features=self.cat_feats, use_best_model=True, eval_set=[(X_val, y_val)],
                               verbose=False,
                               early_stopping_rounds=100)
            except:
                if self.static_data['horizon_type'] == 'multi-output':
                    self.model = CatBoostRegressor(thread_count=self.n_jobs, allow_writing_files=False)
                    self.model.set_params(**self.best_params)
                    self.model.fit(X_train, y_train + pd.DataFrame(np.random.uniform(0, 0.0001, list(y_train.shape)),
                                                                   index=y_train.index, columns=y_train.columns),
                                   cat_features=self.cat_feats,
                                   use_best_model=True,
                                   eval_set=[(X_val, y_val + pd.DataFrame(np.random.uniform(0, 0.0001, list(y_val.shape)),
                                                                   index=y_val.index, columns=y_val.columns))],
                                   verbose=False,
                                   early_stopping_rounds=100)

                else:
                    raise ValueError('Cannot fit Catboost')
            y_pred = self.model.predict(Pool(X_test, cat_features=self.cat_feats))

        y_test = y_test.values
        if len(y_pred.shape) != len(y_test.shape):
            y_test = y_test.ravel()
        norm_val = self.get_rated(y_test)
        self.best_mae_test = np.mean(np.abs(y_pred - y_test) / norm_val)
        self.is_trained = True
        self.save()
        return np.sum(np.square((y_pred - y_test) / norm_val))

    def predict(self, cluster_dates=None):
        X, metadata = self.load_data()
        if isinstance(X, dict):
            for key in sorted(X.keys()):
                X[key].columns = [f'{key}_{col}' for col in X[key].columns]
            X_ = pd.concat([X[key] for key in sorted(X.keys())], axis=1)
            X = X_.copy()
        X[self.cat_feats] = X[self.cat_feats].astype('int')
        cluster_dates = metadata['dates'] if cluster_dates is None else cluster_dates.intersection(metadata['dates'])
        X, _ = self.get_slice(X, cluster_dates, metadata)
        cols = [f'{self.method}_{col}' for col in self.static_data['target_variable']['columns']]

        if cluster_dates.shape[0] > 0:
            if self.method == 'CatBoost':
                return pd.DataFrame(self.model.predict(Pool(X, cat_features=self.cat_feats)), index=cluster_dates, columns=cols)
            else:
                return pd.DataFrame(self.model.predict(X.values), index=cluster_dates, columns=cols)
        else:
            return pd.DataFrame(columns=cols)

    def load(self):
        if os.path.exists(os.path.join(self.path_weights, 'net_weights.pickle')):
            try:
                tmp_dict = joblib.load(os.path.join(self.path_weights, 'net_weights.pickle'))
                self.__dict__.update(tmp_dict)
            except:
                raise ImportError('Cannot load weights for cnn model' + self.path_weights)
        else:
            raise ImportError('Cannot load weights for cnn model' + self.path_weights)

    def save(self):
        tmp_dict = {}
        for k in self.__dict__.keys():
            if k not in ['static_data', 'path_weights', 'refit']:
                tmp_dict[k] = self.__dict__[k]
        joblib.dump(tmp_dict, os.path.join(self.path_weights, 'net_weights.pickle'))
