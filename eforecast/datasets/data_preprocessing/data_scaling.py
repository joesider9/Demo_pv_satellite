import os
import joblib

import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import StandardScaler


class Scaler:
    def __init__(self, static_data, recreate=False, online=False, train=False):
        self.scalers = dict()
        self.online = online
        self.train = train
        self.filename = os.path.join(static_data['path_model'], 'scalers.pickle')
        if os.path.exists(self.filename):
            self.scalers = joblib.load(self.filename)
        if recreate:
            self.scalers = dict()
            if os.path.exists(self.filename):
                os.remove(self.filename)
        self.static_data = static_data
        self.nwp_data_merge = self.static_data['nwp_data_merge']
        self.nwp_data_compress = self.static_data['compress_data']

    def save(self):
        joblib.dump(self.scalers, self.filename)

    def fit(self, data, dataset_name):
        if f'scaler_{dataset_name}' not in self.scalers.keys():
            method = dataset_name.split('_')[-1]
            method = method if method in {'minmax', 'standard', 'maxabs'} else 'minmax'
            scaler = None
            if method == 'minmax':
                scaler = MinMaxScaler()
            elif method == 'standard':
                scaler = StandardScaler()
            elif method == 'maxabs':
                scaler = MaxAbsScaler()
            else:
                raise ValueError(f'Unknown scaling method {method}')
            shape = data.shape
            train_split_index = int(0.7 * shape[0])
            if isinstance(data, pd.Series):
                data = data.to_frame()
            if isinstance(data, pd.DataFrame):
                data = data.values
            if len(shape) > 2:
                data = data.reshape(-1, np.prod(shape[1:]))
            scaler.fit(data[:train_split_index])
            self.scalers[f'scaler_{dataset_name}'] = scaler
            self.save()

    def transform(self, data, dataset_name):
        if f'scaler_{dataset_name}' not in self.scalers.keys() and self.train:
            self.update(data, dataset_name)
        scaler = self.scalers[f'scaler_{dataset_name}']
        shape = data.shape
        index = None
        columns = None
        if isinstance(data, pd.Series):
            data = data.to_frame()
        if isinstance(data, pd.DataFrame):
            index = data.index
            columns = data.columns
            data = data.values
        if len(shape) > 2:
            data = data.reshape(-1, np.prod(shape[1:]))
        data_transformed = scaler.transform(data)
        if len(shape) > 2:
            data_transformed = data_transformed.reshape(-1, *shape[1:])
        if index is not None and columns is not None:
            data_transformed = pd.DataFrame(data_transformed, index=index, columns=columns).astype('float32')
        return data_transformed

    def update(self, data, dataset_name):
        scaler_name = f'scaler_{dataset_name}'
        if scaler_name not in self.scalers.keys():
            if not self.online:
                print(f"Update scaler data {dataset_name}")
                self.fit(data, dataset_name)
                self.save()
            else:
                raise ValueError(f"Scaler named {scaler_name} isn't trained")

    def inverse_transform_data(self, data, dataset_name):
        if f'scaler_{dataset_name}' not in self.scalers.keys() and self.train:
            self.update(data, dataset_name)
        scaler = self.scalers[f'scaler_{dataset_name}']
        shape = data.shape
        index = None
        columns = None
        if isinstance(data, pd.Series):
            data = data.to_frame()
        if isinstance(data, pd.DataFrame):
            index = data.index
            columns = data.columns
            data = data.values
        if len(shape) > 2:
            data = data.reshape(-1, np.prod(shape[1:]))
        data_transformed = scaler.inverse_transform(data)
        if len(shape) > 2:
            data_transformed = data_transformed.reshape(-1, *shape[1:])
        if index is not None and columns is not None:
            data_transformed = pd.DataFrame(data_transformed, index=index, columns=columns).astype('float32')
        return data_transformed
