import os
import numpy as np
import pandas as pd
import joblib

from eforecast.datasets.nwp_data.dataset_nwp_organizer import DatasetNWPsOrganizer
from eforecast.datasets.nwp_data.dataset_nwp_compressor import DatasetNWPsCompressor
from eforecast.datasets.data_preprocessing.data_scaling import Scaler
from eforecast.datasets.data_preprocessing.data_imputing import DataImputer
from eforecast.datasets.files_manager import FilesManager

from eforecast.common_utils.dataset_utils import upsample_tensor
from eforecast.common_utils.dataset_utils import downsample_tensor


class DataPipeline:
    def __init__(self, static_data, recreate=False, online=False, train=False):
        self.train = train
        self.online = online
        self.static_data = static_data
        self.scale_target_method = self.static_data['scale_target_method']
        self.scale_row_method = self.static_data['scale_row_method']
        self.scale_nwp_method = self.static_data['scale_nwp_method']
        self.scaler = Scaler(static_data, recreate=recreate, online=online, train=train)
        if not self.online:
            self.imputer = DataImputer(static_data, recreate=recreate, online=online, train=train)
        else:
            self.imputer = None
        self.nwp_data_merge = self.static_data['nwp_data_merge']
        self.nwp_data_compress = self.static_data['compress_data']
        self.files_manager = FilesManager(static_data, is_online=online, train=train)

    def transform_pipe(self, data, dataset_name, data_dates=None, inverse=False):
        if not inverse:
            data = self.scaler.transform(data, dataset_name)
        if not self.online:
            if data_dates is not None:
                data, new_dates = self.imputer.transform(data, data_dates=data_dates)
            else:
                data, new_dates = self.imputer.transform(data)
        else:
            if data_dates is not None:
                new_dates = data_dates
            else:
                new_dates = data.index
        return data, new_dates


    def merge_nwp_dataset(self, nwp_data, merge_type=None, compress_type=None):
        if merge_type is not None:
            nwp_data_organizer = DatasetNWPsOrganizer(self.static_data, nwp_data)
            nwp_data_merged, nwp_metadata = nwp_data_organizer.merge(merge_type)
            if compress_type is not None:
                nwp_data_compressor = DatasetNWPsCompressor(self.static_data, nwp_data_merged, nwp_metadata,
                                                            compress_type)
                nwp_compressed_all, nwp_compressed_dict = nwp_data_compressor.compress()
                nwp_metadata = {'dates': nwp_compressed_all.index, 'groups': list(nwp_compressed_dict.keys())}
                if merge_type == 'all':
                    if self.static_data['ts_resolution'] == '15min':
                        nwp_compressed_all, dates_new = upsample_tensor(nwp_compressed_all, nwp_compressed_all.index)
                        nwp_metadata['dates'] = dates_new
                    elif self.static_data['ts_resolution'] == 'D':
                        nwp_compressed_all, dates_new = downsample_tensor(nwp_compressed_all, nwp_compressed_all.index,
                                                               resolution=self.static_data['ts_resolution'])
                        nwp_metadata['dates'] = dates_new
                    return nwp_compressed_all, nwp_metadata
                else:
                    if self.static_data['ts_resolution'] == '15min':
                        nwp_compressed_dict, dates_new = upsample_tensor(nwp_compressed_dict, nwp_metadata['dates'])
                        nwp_metadata['dates'] = dates_new
                    elif self.static_data['ts_resolution'] == 'D':
                        nwp_compressed_dict, dates_new = downsample_tensor(nwp_compressed_dict, nwp_metadata['dates'],
                                                               resolution=self.static_data['ts_resolution'])
                        nwp_metadata['dates'] = dates_new
                    return nwp_compressed_dict, nwp_metadata
            else:
                if self.static_data['ts_resolution'] == '15min':
                    nwp_data_merged, dates_new = upsample_tensor(nwp_data_merged, nwp_metadata['dates'])
                    nwp_metadata['dates'] = dates_new
                elif self.static_data['ts_resolution'] == 'D':
                    nwp_data_merged, dates_new = downsample_tensor(nwp_data_merged, nwp_metadata['dates'],
                                                                       resolution=self.static_data['ts_resolution'])
                    nwp_metadata['dates'] = dates_new
                return nwp_data_merged, nwp_metadata
        else:
            return None, None


    def fit_pipe(self):
        if self.train:
            self.fit_row_data_pipe()
            self.fit_lstm_data_pipe()
            self.fit_nwp_data_pipe()

    def fit_row_data_pipe(self):
        target = self.files_manager.check_if_exists_target()
        if target is None:
            raise ImportError(f'Cannot find target dataset')
        print(f"Fit {self.scale_target_method} scaler for target data")
        dataset_name = f'target_{self.scale_target_method}'
        self.scaler.fit(target, dataset_name)

        data_row = self.files_manager.check_if_exists_row_data()
        if data_row is not None:
            for what_data, data in data_row.items():
                if data is None:
                    continue
                for scale_method in self.scale_row_method:
                    print(f"Fit {scale_method} scaler for row {what_data} data")
                    dataset_name = f'data_row_{scale_method}_{what_data}'
                    self.scaler.fit(data, dataset_name)
                    print(f"Fit imputer for data row {what_data}")
                    self.imputer.fit(data)
                    print(f"Fit sorter for data row {what_data}")
        self.save()

    def fit_nwp_data_pipe(self):
        data_nwp = self.files_manager.check_if_exists_nwp_data()
        if data_nwp is not None:
            for method in self.scale_nwp_method:
                for area, area_data in data_nwp.items():
                    for variable, var_data in area_data.items():
                        for vendor, nwp_provide_data in var_data.items():
                            data = nwp_provide_data['data'].copy()
                            dataset_name = f'nwp_{area}_{variable}_{vendor}_{method}'
                            print(f"Fit scaler for {dataset_name}")
                            self.scaler.fit(data, dataset_name)
                            data = self.scaler.transform(data, dataset_name)
                            print(f"Fit imputer for {dataset_name}")
                            self.imputer.fit(data, data_dates=nwp_provide_data['dates'])

    def fit_lstm_data_pipe(self):
        data_lstm_dict = self.files_manager.check_if_exists_lstm_data()
        if data_lstm_dict is not None:
            data_lstm = data_lstm_dict['data']
            metadata = data_lstm_dict['metadata']
            for method in self.scale_row_method:
                for time_type, value_dicts in data_lstm.items():
                    for var_name, value_df in value_dicts.items():
                        data = value_df.copy()
                        dataset_name = f'lstm_{time_type}_{var_name}_{method}'
                        print(f"Fit scaler for {dataset_name}")
                        self.scaler.fit(data, dataset_name)
                        print(f"Fit imputer for {dataset_name}")
                        self.imputer.fit(data, data_dates=metadata['dates'])


    def save(self):
        self.scaler.save()
        self.imputer.save()
