import os
import joblib

import numpy as np
import pandas as pd

from sklearn.impute import MissingIndicator

from eforecast.datasets.files_manager import FilesManager


class DataImputer:
    def __init__(self, static_data, recreate=False, online=False, train=False):
        self.imputers = dict()
        self.online = online
        self.train = train
        self.filename = os.path.join(static_data['path_model'], 'imputers.pickle')
        if os.path.exists(self.filename):
            try:
                self.imputers = set(joblib.load(self.filename))
            except:
                self.imputers = set()
        if recreate:
            self.imputers = set()
            if os.path.exists(self.filename):
                os.remove(self.filename)
        self.static_data = static_data
        self.nwp_data_merge = self.static_data['nwp_data_merge']
        self.nwp_data_compress = self.static_data['compress_data']
        self.files_manager = FilesManager(static_data, is_online=online)

    def save(self):
        joblib.dump(self.imputers, self.filename)

    def fit(self, data_fit, data_dates=None):
        if isinstance(data_fit, dict):
            for key, values in data_fit.items():
                self._fit(values, data_dates=data_dates)
        else:
            self._fit(data_fit, data_dates=data_dates)

    def _fit(self, data_fit1, data_dates=None):
        columns2 = None
        index2 = None
        missing_indicator = MissingIndicator(features="all")
        shape = data_fit1.shape
        if isinstance(data_fit1, pd.Series):
            data_fit2 = data_fit1.to_frame()
        if isinstance(data_fit1, pd.DataFrame):
            index2 = data_fit1.index
            columns2 = data_fit1.columns
            data_fit2 = data_fit1.values
        if len(shape) > 2:
            index2 = data_dates
            data_fit2 = data_fit1.reshape(-1, np.prod(shape[1:]))
        if index2 is None:
            data_fit2 = data_fit1
            index2 = data_dates
        columns2 = [f'x_{i}' for i in range(data_fit2.shape[1])] if columns2 is None else columns2
        flag_missing = missing_indicator.fit_transform(data_fit2)
        ind_nan_feature = np.where(np.all(flag_missing, axis=0))[0]
        if len(ind_nan_feature) > 0:
            raise ValueError(f'the feature {columns2[ind_nan_feature]} have NaN all their values')
        ind_nan_dates = np.where(np.any(flag_missing, axis=1))[0]
        if len(ind_nan_dates) > 0:
            self.imputers = self.imputers.union(set(index2[ind_nan_dates]))
        self.save()

    def transform(self, data, data_dates=None):
        if isinstance(data, dict):
            data_transform = dict()
            for key, values in data.items():
                data_tr, index1 = self._transform(values, data_dates=data_dates)
                data_transform[key] = data_tr
            new_dates = index1
        else:
            data_transform, new_dates = self._transform(data, data_dates=data_dates)
        return data_transform.astype('float32'), new_dates

    def _transform(self, data, data_dates=None):
        self.update(data, data_dates=data_dates)
        index = None
        columns = None
        if isinstance(data, pd.Series):
            data = data.to_frame()
        if isinstance(data, pd.DataFrame):
            index = data.index
            columns = data.columns
            data = data.values
        if index is None and data_dates is not None:
            index = data_dates
        if index is None:
            raise ValueError('You should provide dates of numpy array')

        dates = pd.DatetimeIndex([d for d in index if d not in self.imputers])
        ind_dates = index.get_indexer(dates)
        data_transformed = data[ind_dates]
        index = index[ind_dates]

        if index is not None and columns is not None:
            data_transformed = pd.DataFrame(data_transformed, index=dates, columns=columns)
        return data_transformed.astype('float32'), dates

    def update(self, data_up, data_dates=None):
        if not self.online:
            self.fit(data_up, data_dates=data_dates)
