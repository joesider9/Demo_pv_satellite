import copy

import numpy as np

from sklearn.preprocessing import LabelEncoder

from eforecast.datasets.data_preprocessing.data_imputing import DataImputer
from eforecast.datasets.data_feeder import DataFeeder
from eforecast.datasets.data_preprocessing.data_scaling import Scaler
from eforecast.common_utils.dataset_utils import sync_datasets
from eforecast.common_utils.dataset_utils import sync_data_row_with_tensors
from eforecast.datasets.data_resampling.nwp_resampling import NWPResampler
from eforecast.datasets.data_resampling.row_resampling import RowResampler
from eforecast.datasets.data_resampling.target_resampling import TargetResampler
from eforecast.common_utils.dataset_utils import get_slice
from eforecast.datasets.files_manager import FilesManager



class DataSampler(object):
    def __init__(self, static_data):
        self.static_data = static_data
        self.scale_method = self.static_data['combining']['data_type']['scaling']
        self.merge = self.static_data['combining']['data_type']['merge']
        self.compress = self.static_data['combining']['data_type']['compress']
        self.what_data = self.static_data['combining']['data_type']['what_data']
        self.scaler = Scaler(static_data, recreate=False, online=False, train=True)
        self.scale_row_method = self.static_data['scale_row_method']
        self.problem_type = self.static_data['type']
        self.n_jobs = self.static_data['n_jobs']
        self.imputer = DataImputer(static_data, train=True)
        self.files_manager = FilesManager(self.static_data, is_online=False, train=True)


    def resample_data(self):
        if self.static_data['type'] in {'load', 'fa', 'pv', 'wind'}:
            print('No data_resampling needed for load or fa')
            return
        if self.nwp_models[0]['model'] is not None:
            nwp_data = self.files_manager.check_if_exists_nwp_data()
            nwp_data_sampled = self.files_manager.check_if_exists_nwp_data(resampling=True)
            if nwp_data_sampled is None:
                data_sampler = DataSampler(self.static_data)
                nwp_data_sampled = data_sampler.sampling(nwp_data, dataset_type='nwp')
                self.files_manager.save_nwps(nwp_data_sampled, resampling=True)

        data_row = self.files_manager.check_if_exists_row_data()
        data_row_sampled = self.files_manager.check_if_exists_row_data(resampling=True)
        if data_row_sampled is None:
            data_sampler = DataSampler(self.static_data)
            if self.static_data['type'] in {'load', 'fa'}:
                data_lstm_dict = self.files_manager.check_if_exists_lstm_data()
                data_row_sampled, X_lstm_dict = data_sampler.sampling(data_row, dataset_type='row',
                                                                      X_lstm_dict=data_lstm_dict)
                self.files_manager.save_lstm_data(X_lstm_dict['data'], X_lstm_dict['metadata'], resampling=True)
            else:
                data_row_sampled = data_sampler.sampling(data_row, dataset_type='row')
            self.files_manager.save_row_data(data_row_sampled, resampling=True)
        target = self.files_manager.check_if_exists_target()
        target_sampled = self.files_manager.check_if_exists_target(resampling=True)
        if target_sampled is None:
            data_sampler = DataSampler(self.static_data)
            target_sampled = data_sampler.sampling(target, dataset_type='target')
            self.files_manager.save_target(target_sampled, resampling=True)

    def imblearn_nwp(self, dataset, variable_name, random_state=42):
        X_3d = dataset['data']
        dates = dataset['dates']
        X_3d, dates = self.split(X_3d, dates)
        X_3d, dates = self.imputer.transform(X_3d, data_dates=dates)
        if variable_name in {'Flux', 'Temperature'}:
            y = dates.hour.values
        else:
            y = np.zeros(X_3d.shape[0]).astype('int')

        strategy = "all"

        if np.any(np.bincount(y.ravel()) < 2):
            raise ValueError('Very small sample, cannot perform data_resampling')
        sm = NWPResampler(sampling_strategy=strategy, random_state=random_state, n_jobs=self.n_jobs)

        print(f'Start data_resampling for nwp variable {variable_name}')
        dataset['data'] = sm.fit_resample(X_3d, y.ravel())
        dataset['dates'] = dates
        return dataset

    def sampling_nwps(self, nwp_data):
        nwp_data_sampled = copy.deepcopy(nwp_data)
        for area, area_data in nwp_data.items():
            for variable, var_data in area_data.items():
                for vendor, nwp_provide_data in var_data.items():
                    nwp_data_sampled[area][variable][vendor] = self.imblearn_nwp(nwp_provide_data, variable)
        return nwp_data_sampled

    def feed_data(self):
        print('Read data for Sampling....')
        data_feeder = DataFeeder(self.static_data, train=True)
        data, _ = data_feeder.feed_inputs(merge=self.merge, compress=self.compress,
                                          scale_nwp_method=self.scale_method,
                                          data_tag=self.what_data, inverse_transform=True)
        data_feeder_resampled = DataFeeder(self.static_data, train=True, resampling=True)
        data_resampled, _ = data_feeder_resampled.feed_inputs(merge=self.merge, compress=self.compress,
                                                              scale_nwp_method=self.scale_method,
                                                              data_tag=self.what_data, inverse_transform=True)
        return sync_datasets(data, data_resampled, name1='real_data', name2='resampled_data')

    def get_row_labels(self, dataset):
        CategoricalFeatures = ['dayweek', 'hour', 'month', 'sp_index']
        DateFeatures = ['azimuth', 'zenith']
        data, dates = self.imputer.transform(dataset)
        data, dates = self.split(data, dates)
        cat_feats = [v_name for v_name in data.columns
                     for c_feats in CategoricalFeatures if c_feats in v_name
                              and len(np.unique(data[v_name].values)) < 30]
        date_feats = [v_name for v_name in data.columns
                      for c_feats in DateFeatures if c_feats in v_name]
        if self.problem_type in {'load', 'fa'}:
            if 'sp_index' not in data.columns:
                raise ValueError("sp_index variable is not exists in dataset for type load or fa")
            sp_index_flag = True
        else:
            sp_index_flag = False
        if self.problem_type in {'load', 'pv'}:
            if 'hour' not in data.columns:
                raise ValueError("hour variable is not exists in dataset for type load or pv")
            hour_flag = True
        else:
            hour_flag = False
        if hour_flag and sp_index_flag:
            encodes = (data['hour'].astype('str') + data['sp_index'].astype('str')).values
            labels = LabelEncoder().fit_transform(encodes).astype('int')
        elif not hour_flag and sp_index_flag:
            labels = LabelEncoder().fit_transform(data['sp_index'].astype('str').values).astype('int')
        elif hour_flag and not sp_index_flag:
            labels = data['hour'].values.astype('int')
        else:
            labels = np.zeros(data.shape[0]).astype('int')
        return data, labels, cat_feats, date_feats

    def sampling_row(self, dataset, X_lstm_dict=None):
        if X_lstm_dict is not None:
            metadata_lstm = X_lstm_dict['metadata']
            X_lstm = X_lstm_dict['data']
            index = dataset.index
            X_lstm, dataset, dates = sync_data_row_with_tensors(data_tensor=X_lstm, dates_tensor=metadata_lstm['dates'],
                                                                data_row=dataset, dates_row=index)
            dataset_name = f'lstm_{self.scale_method}'
            X_lstm = self.scaler.transform(X_lstm, dataset_name)
            metadata_lstm['dates'] = dates

        data, labels, cat_feats, date_feats = self.get_row_labels(dataset)
        dataset_name = f'data_row_{self.scale_row_method}'
        data = self.scaler.transform(data, dataset_name)
        if X_lstm_dict is not None:
            num_feats_lstm = []
            for i, variable in enumerate(metadata_lstm['variables']):
                flag = False
                for cat_feat in cat_feats + date_feats + ['load']:
                    if variable in cat_feat:
                        flag = True
                if not flag:
                    num_feats_lstm.append(i)
            X_lstm_num = X_lstm[:, :, num_feats_lstm]
        else:
            num_feats_lstm = None
            X_lstm = None
            X_lstm_num = None
        cols_for_sampling = [col for col in data.columns if col not in cat_feats + date_feats and 'load' not in col]
        if len(cols_for_sampling) > 0:
            if np.any(np.bincount(labels.ravel()) < 2):
                raise ValueError('Very small sample, cannot perform data_resampling')
            sm = RowResampler(sampling_strategy='all', random_state=42, n_jobs=self.n_jobs)

            print(f'Start data_resampling for row data')
            if X_lstm_num is None:
                data_sampled = data.copy(deep=True)
                data_sampled[cols_for_sampling] = sm.fit_resample(data[cols_for_sampling], labels.ravel())
                dataset_name = f'data_row_{self.scale_row_method}'
                data_sampled = self.scaler.inverse_transform_data(data_sampled, dataset_name)
                return data_sampled
            else:
                data_sampled = data.copy(deep=True)
                if X_lstm_num.shape[-1] > 0:
                    data_sampled_, X_lstm_resampled_ = sm.fit_resample(data[cols_for_sampling], labels.ravel(),
                                                                       X_lstm=X_lstm_num)
                else:
                    data_sampled_ = sm.fit_resample(data[cols_for_sampling], labels.ravel())
                    X_lstm_resampled_ = X_lstm_num
                data_sampled[cols_for_sampling] = data_sampled_
                dataset_name = f'data_row_{self.scale_row_method}'
                data_sampled = self.scaler.inverse_transform_data(data_sampled, dataset_name)
                X_lstm_resampled = copy.deepcopy(X_lstm)
                X_lstm_resampled[:, :, num_feats_lstm] = X_lstm_resampled_
                dataset_name = f'lstm_{self.scale_method}'
                X_lstm_resampled = self.scaler.inverse_transform_data(X_lstm_resampled, dataset_name)
                X_lstm_dict['data'] = X_lstm_resampled
                X_lstm_dict['metadata'] = metadata_lstm
                return data_sampled, X_lstm_dict
        else:
            return data

    def sampling_target(self, target):
        data, data_resampled = self.feed_data()
        target, data = sync_datasets(target, data, name1='target', name2='real_data')
        target, data_resampled = sync_datasets(target, data_resampled, name1='target', name2='resampled_data')
        data, labels, cat_feats, date_feats = self.get_row_labels(data)
        if np.any(np.bincount(labels.ravel()) < 2):
            raise ValueError('Very small sample, cannot perform data_resampling')
        sm = TargetResampler(sampling_strategy='all', random_state=42, n_jobs=self.n_jobs)

        print(f'Start data_resampling for row data')
        target_sampled = sm.fit_resample(target, data, data_resampled, labels.ravel())
        return target_sampled

    def sampling(self, dataset, dataset_type=None, X_lstm_dict=None):
        if dataset_type == 'nwp':
            return self.sampling_nwps(dataset)
        elif dataset_type == 'row':
            return self.sampling_row(dataset, X_lstm_dict=X_lstm_dict)
        elif dataset_type == 'target':
            dataset = dataset.dropna(axis='index')
            return self.sampling_target(dataset)

    def split(self, data, dates):
        ind = np.where(dates <= self.static_data['Evaluation_start'])[0]
        return get_slice(data, ind), dates[ind]
