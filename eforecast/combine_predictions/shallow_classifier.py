import os
import joblib

import numpy as np
import pandas as pd

from catboost import CatBoostClassifier
from catboost import Pool

from sklearn.metrics import log_loss
from sklearn.metrics import roc_auc_score
from eforecast.datasets.data_feeder import DataFeeder
from eforecast.common_utils.dataset_utils import sync_datasets
from eforecast.common_utils.dataset_utils import sync_dict_df

pd.set_option('display.expand_frame_repr', False)
from sklearn.model_selection import train_test_split

CategoricalFeatures = ['dayweek', 'hour', 'month', 'sp_index']


class ShallowModelClassifier:
    def __init__(self, static_data, path_weights, predictors=None, params=None, n_jobs=1, refit=False,
                 train=False, online=False):
        self.best_mae_val = None
        self.best_mae_test = None
        self.is_trained = False
        self.refit = refit
        self.static_data = static_data
        self.rated = static_data['rated']
        self.n_jobs = n_jobs
        self.predictors = predictors
        if params is not None:
            self.params = params
            self.method = self.params['method']
            self.cluster_name = self.params['cluster_name']
            self.merge = self.params['merge']
            self.compress = self.params['compress']
            self.scale_nwp_method = self.params['scale_nwp_method']
            self.scale_row_method = self.params['scale_row_method']
            if self.method == 'CatBoost':
                self.model = CatBoostClassifier()
            else:
                raise ValueError(f'Unknown method {self.method} for shallow models')
            self.best_params = {'iterations': 1000,
                                'learning_rate': 0.005,
                                'l2_leaf_reg': 1,
                                "objective": "RMSE",
                                'min_data_in_leaf': 2,
                                "depth": 1,
                                "boosting_type": "Ordered",
                                "bootstrap_type": "Bayesian",
                                "eval_metric": "MAE"}
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

    def load_data(self, y=None):
        what_data = self.params['what_data']
        x, mdata = self.data_feeder.feed_inputs(what_data, merge=self.params['merge'], compress=self.params['compress'],
                                                scale_nwp_method=self.params['scale_nwp_method'],
                                                scale_row_method=self.params['scale_row_method'],
                                                feature_selection_method=self.params['feature_selection_method'],
                                                cluster={'cluster_name': self.params['cluster_name'],
                                                         'cluster_path': self.params['cluster_dir']})
        X = x[what_data]
        metadata = mdata[what_data]
        if y is None:
            return X, metadata
        else:
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

    def fit(self, best_predictors, cv_masks):
        X, y, metadata = self.load_data(y=best_predictors)
        if isinstance(X, dict):
            for key in sorted(X.keys()):
                X[key].columns = [f'{key}_{col}' for col in X[key].columns]
            X_ = pd.concat([X[key] for key in sorted(X.keys())], axis=1)
            X = X_.copy()
        self.cat_feats = list(set([v_name for v_name in X.columns if len(np.unique(X[v_name].values)) < 30]))

        X[self.cat_feats] = X[self.cat_feats].astype('int')
        split_test = int(X.shape[0] * (1 - self.static_data['val_test_ratio']))
        mask_test = X.index[split_test:]
        mask_train = X.index[:split_test]
        y_temp = y.iloc[:split_test]

        mask_train, mask_val = train_test_split(mask_train, test_size=self.static_data['val_test_ratio'],
                                                random_state=42,
                                                stratify=y_temp.loc[mask_train].values)
        cv_masks = [mask_train, mask_test, mask_val]
        X_train, y_train = self.get_slice(X, cv_masks[0], metadata, y=y)
        X_val, y_val = self.get_slice(X, cv_masks[1], metadata, y=y)
        X_test, y_test = self.get_slice(X, cv_masks[2], metadata, y=y)

        if self.method in {'CatBoost'}:
            self.model.fit(X_train, y_train, cat_features=self.cat_feats, use_best_model=True, eval_set=[(X_val, y_val)],
                           verbose=False,
                           early_stopping_rounds=30)
            y_pred = self.model.predict_proba(Pool(X_test, cat_features=self.cat_feats))
        else:
            raise ValueError(f'Unknown method {self.method} for shallow models')

        y_test = y_test.values
        if len(self.model.classes_) == 2:
            self.best_mae_test = roc_auc_score(y_test.ravel(), y_pred[:, 1])
        else:
            self.best_mae_test = roc_auc_score(y_test.ravel(), y_pred, multi_class='ovr')
        self.is_trained = True
        self.save()
        return self.best_mae_test

    def predict_proba(self, cluster_dates=None):
        X, metadata = self.load_data(y=None)
        if isinstance(X, dict):
            for key in sorted(X.keys()):
                X[key].columns = [f'{key}_{col}' for col in X[key].columns]
            X_ = pd.concat([X[key] for key in sorted(X.keys())], axis=1)
            X = X_.copy()
        X[self.cat_feats] = X[self.cat_feats].astype('int')
        cluster_dates = metadata['dates'] if cluster_dates is None else cluster_dates.intersection(metadata['dates'])
        X, _ = self.get_slice(X, cluster_dates, metadata)
        if self.method == 'CatBoost':
            return self.model.predict_proba(Pool(X, cat_features=self.cat_feats))
        else:
            raise ValueError(f'Unknown method {self.method} for shallow models')

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
