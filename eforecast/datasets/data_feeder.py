import copy
import os

import numpy as np
import pandas as pd

from eforecast.common_utils.dataset_utils import concat_pandas_df
from eforecast.common_utils.dataset_utils import concat_df_dict
from eforecast.common_utils.dataset_utils import concat_dict_dict
from eforecast.common_utils.dataset_utils import get_dates_from_dict

from eforecast.datasets.files_manager import FilesManager
from eforecast.datasets.data_preprocessing.data_pipeline import DataPipeline
from eforecast.feature_selection.feature_selection_transform import FeatureSelector


class DataFeeder:
    def __init__(self, static_data, online=False, train=False):
        self.static_data = static_data
        self.online = online
        self.train = train
        self.scale_target_method = self.static_data['scale_target_method']
        self.scale_row_method = self.static_data['scale_row_method']
        self.scale_nwp_method = self.static_data['scale_nwp_method']
        self.calendar_variables = [var_data['name'] for var_data in self.static_data['variables']
                                   if var_data['type'] == 'calendar']
        self.dataset_tags = set([key for net in self.static_data['experiments'].values() 
                                 for key in net.keys() 
                                 if key not in {'output', 'hidden_layer'}]).union({'row_all'})
        self.files_manager = FilesManager(static_data, is_online=online, train=self.train)
        self.pipeline = DataPipeline(self.static_data, online=online, train=self.train)
        self.feature_selector = FeatureSelector(self.static_data, online=online, train=self.train)

    def load_nwp_data(self, scale=None, inverse=False, get_all=False):
        data_nwp = self.files_manager.check_if_exists_nwp_data(get_all=get_all)
        if scale is None:
            return data_nwp
        if data_nwp is None:
            raise ImportError(f'Cannot find nwp dataset')
        else:
            for area, area_data in data_nwp.items():
                for variable, var_data in area_data.items():
                    for vendor, nwp_provide_data in var_data.items():
                        data = nwp_provide_data['data'].copy()
                        dataset_name = f'nwp_{area}_{variable}_{vendor}_{scale}'
                        data_nwp[area][variable][vendor]['data'], dates = self.pipeline.transform_pipe(data, dataset_name,
                                                                                        data_dates=
                                                                                        nwp_provide_data['dates'],
                                                                                        inverse=inverse)
                        data_nwp[area][variable][vendor]['dates'] = dates
        return data_nwp

    def load_images(self, scale=None, inverse=False):
        data_image = self.files_manager.check_if_exists_image_data()
        if data_image is None:
            raise ImportError(f'Cannot find image dataset')
        return data_image

    def load_lstm(self, scale=None, inverse_data=False, inverse_calendar=False):
        calendar_vars = {'month': 12, 'day': 30, 'dayweek': 7, 'hour': 24, 'minute': 60}
        data_lstm = self.files_manager.check_if_exists_lstm_data()
        if data_lstm['data'] is None:
            raise ImportError(f'Cannot find lstm dataset')
        else:
            for time_type, value_dicts in data_lstm['data'].items():
                for var_name, value_df in value_dicts.items():
                    data = value_df.copy()
                    dataset_name = f'lstm_{time_type}_{var_name}_{scale}'
                    inverse = inverse_data or inverse_calendar if var_name in calendar_vars.keys() \
                        else inverse_data
                    data_lstm['data'][time_type][var_name], dates = self.pipeline.transform_pipe(data, dataset_name,
                                                                                  data_dates=
                                                                                  data_lstm['metadata']['dates'],
                                                                                  inverse=inverse)
                    data_lstm['metadata']['dates'] = dates
        return data_lstm

    def load_row_data(self, scale=None, inverse_data=False, inverse_calendar=False):
        data_row = self.files_manager.check_if_exists_row_data()
        if data_row is None:
            raise ImportError(f'Cannot find row dataset')
        else:
            dataset_name = f'data_row_{scale}_row_obs'
            if data_row['row_obs'] is not None:
                data_row['row_obs'], _= self.pipeline.transform_pipe(data_row['row_obs'],
                                                                     dataset_name, inverse=inverse_data)
            dataset_name = f'data_row_{scale}_calendar'
            if data_row['calendar'] is not None:
                data_row['calendar'], _= self.pipeline.transform_pipe(data_row['calendar'],
                                                                      dataset_name, inverse=inverse_calendar)
        return data_row

    def load_target(self, inverse=False):
        target = None
        if not self.online:
            target = self.files_manager.check_if_exists_target()
            if target is None:
                raise ImportError(f'Cannot find target dataset')
            else:
                dataset_name = f'target_{self.scale_target_method}'
                target = self.pipeline.transform_pipe(target, dataset_name, inverse=inverse)
        else:
            raise ImportError(f'Cannot import target dataset when is online True and offline False')
        return target

    def load_dataset(self, data_tag, merge=None, compress=None, scale_row_method=None, scale_nwp_method=None,
                     inverse=False, get_all=False, transform_calendar=False):
        calendar_vars = {'month': 12, 'day': 30, 'dayweek': 7, 'hour': 24, 'minute': 60}
        if data_tag == 'row_data':
            data = self.load_row_data(scale=scale_row_method, inverse_data=inverse,
                                      inverse_calendar=inverse or transform_calendar)
            if transform_calendar:
                dfs = []
                for var in data['calendar'].columns:
                    if var.split('_')[0] in calendar_vars.keys():
                        value1 = data['calendar'][var].to_frame()
                        sin = np.sin((2 * np.pi * value1.values / calendar_vars[var.split('_')[0]]).astype('float'))
                        cos = np.cos((2 * np.pi * value1.values / calendar_vars[var.split('_')[0]]).astype('float'))
                        cal_val = np.concatenate([sin, cos], axis=-1)
                        var_cal = [var.replace(var.split('_')[0], f'{var.split("_")[0]}_sin'),
                                   var.replace(var.split('_')[0], f'{var.split("_")[0]}_cos')]
                        cal_df = pd.DataFrame(cal_val, columns=var_cal, index=value1.index)
                        dfs.append(cal_df)
                    elif 'sp_index' in var:
                        value1 = data['calendar'][var].to_frame() / 100
                        dfs.append(value1)
                    elif var.split('_')[0] == 'dayofyear':
                        value1 = data['calendar'][var].to_frame() / 365
                        dfs.append(value1)
                    else:
                        raise ValueError(f'Unknown calendar variable: {var}')

                data['calendar'] = pd.concat(dfs, axis=1)
            metadata = {'dates': data['calendar'].index}
        elif data_tag == 'nwp':
            data_nwp = self.load_nwp_data(scale=scale_nwp_method, inverse=inverse, get_all=get_all)
            data, metadata = self.pipeline.merge_nwp_dataset(data_nwp, merge_type=merge,
                                                                         compress_type=compress)
        elif data_tag == 'lstm':
            data_lstm = self.load_lstm(scale=scale_row_method, inverse_data=inverse,
                                       inverse_calendar=inverse or transform_calendar)
            data, metadata = data_lstm['data'], data_lstm['metadata']

        elif data_tag == 'images':
            data = self.load_images()
            if not isinstance(data, dict):
                metadata = {'dates': data}
            else:
                raise NotImplementedError('Image dataset is not supported yet. Only dates included.')
        else:
            raise ValueError(f'Unknown Dataset name: {data_tag}. Should be one of row_data, nwp, lstm or images')
        return data, metadata

    def get_row_data(self, what_data, merge=None, compress=None, scale_row_method=None, scale_nwp_method=None,
                     inverse=False, get_all=False, transform_calendar=False):
        data_tags = what_data.split('_')[1:]
        datasets, metadatasets = [], dict()
        for tag in data_tags:
            if tag in {'calendar', 'obs'}:
                data, metadata = self.load_dataset('row_data', scale_row_method=scale_row_method,
                                                   inverse=inverse, transform_calendar=transform_calendar)
                if data is None:
                    continue
                if tag == 'all':
                    datasets.append(pd.concat([data['row_obs'], data['calendar']], axis=1))
                else:
                    datasets.append(data['row_obs'] if tag == 'obs' else data['calendar'])
                metadatasets.update(metadata)
            elif tag == 'nwp':
                data, metadata = self.load_dataset('nwp', merge=merge, compress=compress,
                                                   scale_nwp_method=scale_nwp_method, inverse=inverse, get_all=get_all)
                datasets.append(data)
                metadatasets.update(metadata)
            elif tag == 'all':
                data, metadata = self.load_dataset('row_data', scale_row_method=scale_row_method,
                                                   inverse=inverse, transform_calendar=transform_calendar)
                datasets.append(pd.concat([data['row_obs'], data['calendar']], axis=1))
                metadatasets.update(metadata)
                if self.static_data['NWP'] is not None:
                    data, metadata = self.load_dataset('nwp', merge=merge, compress=compress,
                                                       scale_nwp_method=scale_nwp_method, inverse=inverse)
                    datasets.append(data)
                    metadatasets.update(metadata)
            else:
                raise ValueError(f'Unknown row data tag: {tag}, Should be one of all, calendar, obs or nwp')
        if len(datasets) == 1:
            dataset = datasets[0]
            if isinstance(dataset, pd.DataFrame):
                dates = dataset.index
            else:
                if dataset is None:
                    raise ValueError(f'No {what_data} datasets found. Check experiment input tags')
                dates = get_dates_from_dict(dataset)
        else:
            if len(datasets) == 0:
                raise ValueError('No row datasets found')
            dataset = datasets[0]
            if isinstance(dataset, pd.DataFrame):
                dates = dataset.index
            else:
                if dataset is None:
                    raise ValueError(f'No {what_data} datasets found. Check experiment input tags')
                dates = get_dates_from_dict(dataset)
            for dataset_temp in datasets[1:]:
                if dataset_temp is None:
                    raise ValueError(f'No {what_data} datasets found. Check experiment input tags')
                if isinstance(dataset, pd.DataFrame) and isinstance(dataset_temp, pd.DataFrame):
                    dataset = concat_pandas_df(dataset, dataset_temp)
                    dates = dataset.index
                elif isinstance(dataset, pd.DataFrame) and isinstance(dataset_temp, dict):
                    dataset, dates = concat_df_dict(dataset_temp, dataset)
                elif isinstance(dataset, dict) and isinstance(dataset_temp, pd.DataFrame):
                    dataset, dates = concat_df_dict(dataset, dataset_temp)
                elif isinstance(dataset, dict) and isinstance(dataset_temp, dict):
                    dataset, dates = concat_dict_dict(dataset_temp, dataset)
                else:
                    dates = None
                    raise TypeError(f'Unknown dataset type, {type(dataset)}, {type(dataset_temp)}')
        if isinstance(dataset, pd.DataFrame):
            dataset = dataset.dropna(axis='index', how='any')
            dates = dataset.index
        else:
            for key in dataset.keys():
                dataset[key] = dataset[key].dropna(axis='index', how='any')
                dates = dates.intersection(dataset[key].index)
        metadatasets['dates'] = dates
        return dataset, metadatasets

    def feed_inputs(self, data_tag, merge=None, compress=None, scale_nwp_method=None, scale_row_method=None,
                    feature_selection_method=None, cluster=None, inverse_transform=False,
                    get_all=False, get_lstm_vars=False, transform_calendar=False):

        if data_tag not in self.dataset_tags and compress != 'minimal':
            raise ValueError(f'Unknown dataset tag: {data_tag}')
        if 'row' in data_tag:

            data, metadata = self.get_row_data(data_tag, merge=merge, compress=compress,
                                               scale_row_method=scale_row_method,
                                               scale_nwp_method=scale_nwp_method,
                                               inverse=inverse_transform,
                                               get_all=get_all,
                                               transform_calendar=transform_calendar)
            if feature_selection_method is not None:
                data_type = {data_tag: {'scale_row_method': scale_row_method,
                                        'scale_nwp_method': scale_nwp_method,
                                        'merge': merge,
                                        'compress': compress
                                        }}
                dataset_name = ('_'.join([t for t in data_type[data_tag].values() if t is not None]) +
                                f'_{data_tag}')
                feature_selector_tag =(f'feature_selector_{cluster["cluster_name"]}_{feature_selection_method}_'
                                   f'{dataset_name}')
                data = self.feature_selector.transform(feature_selector_tag, cluster["cluster_path"], data,
                                                       transform_calendar=transform_calendar)
        elif data_tag == 'images':
            data, metadata = self.load_dataset('images')
        elif data_tag == 'nwp':
            data, metadata = self.load_dataset('nwp', merge=merge, scale_nwp_method=scale_nwp_method,
                                               inverse=inverse_transform, get_all=get_all)
        elif data_tag == 'lstm':
            data, metadata = self.load_dataset('lstm', scale_row_method=scale_row_method,
                                               inverse=inverse_transform, transform_calendar=transform_calendar)
            if get_lstm_vars:
                return {data_tag: data}, {data_tag: metadata}
            calendar_vars = {'month': 12, 'day': 30, 'dayweek': 7, 'hour': 24, 'minute': 60}
            if feature_selection_method is not None:
                feature_selector_tag = (f'feature_selector_{cluster["cluster_name"]}_{feature_selection_method}_'
                                       f'{scale_row_method}_lstm')
                data = self.feature_selector.transform(feature_selector_tag, cluster["cluster_path"], data)
            past_obs = []
            past_data = []
            future_data = []
            past_calendar = []
            future_calendar = []
            lstm_vars = []
            cal_vars = []
            data_lstm = dict()
            for key, value in data.items():
                for key1 in sorted(value.keys()):
                    value1 = value[key1]
                    columns_sorted = value1.columns[np.argsort([int(col.split('_lag_')[-1]) for col in value1.columns])]
                    value1 = value1[columns_sorted]
                    if key1 in calendar_vars.keys():
                        if transform_calendar:
                            sin = np.expand_dims(np.sin((2 * np.pi * value1.values / calendar_vars[key1]).astype('float')), axis=-1)
                            cos = np.expand_dims((np.cos(2 * np.pi * value1.values / calendar_vars[key1]).astype('float')), axis=-1)
                            cal_val = np.concatenate([sin, cos], axis=-1)
                            var_cal = [f'{key1}_sin', f'{key1}_cos']
                        else:
                            cal_val = np.expand_dims(value1.values, axis=-1)
                            var_cal = [key1]
                        past_calendar.append(cal_val) if key == 'past' else future_calendar.append(cal_val)
                        if key1 not in cal_vars:
                            cal_vars += var_cal
                    elif key1 in metadata['past_variables']:
                        past_obs.append(np.expand_dims(value1, axis=-1))
                    else:
                        past_data.append(np.expand_dims(value1, axis=-1)) if key == 'past' \
                            else future_data.append(np.expand_dims(value1, axis=-1))
                    lstm_vars.append(key1)
            if len(past_data) > 0:
                data_lstm['past_data'] = np.concatenate(past_data, -1)
            if len(past_obs) > 0:
                data_lstm['past_obs'] =  np.concatenate(past_obs, -1)
            if len(past_calendar) > 0:
                data_lstm['past_calendar'] =  np.concatenate(past_calendar, -1)
            if len(future_data) > 0:
                data_lstm['future_data'] =  np.concatenate(future_data, -1)
            if len(future_calendar) > 0:
                data_lstm['future_calendar'] =  np.concatenate(future_calendar, -1)
            metadata['variables'] = lstm_vars
            metadata['cal_vars'] = cal_vars
            data = copy.deepcopy(data_lstm)


        return {data_tag: data}, {data_tag: metadata}


    def feed_target(self, inverse=False):
        y, _ = self.load_target(inverse=inverse)

        return y
