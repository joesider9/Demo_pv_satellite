import os
import pvlib
import pandas as pd
import numpy as np
from eforecast.datasets.image_data.dataset_img_creator import DatasetImageCreator
from eforecast.datasets.nwp_data.dataset_nwp_creator import DatasetNWPsCreator
from eforecast.datasets.data_transformations import DataTransformer
from eforecast.datasets.files_manager import FilesManager
from eforecast.datasets.data_feeder import DataFeeder

from eforecast.common_utils.date_utils import sp_index
from eforecast.common_utils.date_utils import last_year_lags
from eforecast.common_utils.dataset_utils import fix_timeseries_dates


class DatasetCreator:
    def __init__(self, static_data, recreate=False, train=True, is_online=False, dates=None):
        self.static_data = static_data
        self.is_online = is_online
        self.train = train
        self.dates = dates
        self.data = None
        if ((self.is_online and self.static_data['horizon_type'] == 'multi-output')
                or (self.static_data['type'] in {'load', 'FA'})) or self.train:
            self.load_data()
        if self.dates is None:
            raise ValueError('Cannot find dates')
        self.path_data = self.static_data['path_data']
        self.horizon_type = static_data['horizon_type']
        self.nwp_models = static_data['NWP']
        self.nwp_data_merge = self.static_data['nwp_data_merge']
        self.nwp_data_compress = self.static_data['compress_data']
        self.transformer = DataTransformer(self.static_data, recreate=recreate, is_online=self.is_online,
                                           train=self.train)
        self.files_manager = FilesManager(self.static_data, is_online=self.is_online, train=self.train)
        self.data_feeder = DataFeeder(self.static_data, online=self.is_online, train=self.train)
        if recreate or is_online:
            self.files_manager.remove_row_data_files()
            self.files_manager.remove_lstm_data_files()
            self.files_manager.remove_nwps()
            self.files_manager.remove_images()
            if not is_online:
                self.files_manager.remove_target_files()

    def load_data(self):
        if self.is_online:
            data = pd.read_csv(os.path.join(self.static_data['path_data'], self.static_data['filename']),
                               header=0, index_col=0, parse_dates=True)
        else:
            if not os.path.exists(self.static_data['filename']):
                raise ImportError(f"Cannot find the main file csv {self.static_data['filename']}")
            data = pd.read_csv(self.static_data['filename'], header=0, index_col=0, parse_dates=True)
        data = data.dropna(how='all', axis=0)
        self.data = fix_timeseries_dates(data, self.static_data['ts_resolution'])
        if self.dates is None or not self.is_online:
            self.dates = data.index
        print('Time series imported successfully from the file %s', self.static_data['filename'])

    def create_nwp_dataset(self, parallel):
        if self.nwp_models is not None:
            nwp_data_creator = DatasetNWPsCreator(self.static_data, self.transformer, dates=self.dates,
                                                  is_online=self.is_online, parallel=parallel)
            nwp_data = self.files_manager.check_if_exists_nwp_data()
            if nwp_data is None:
                nwp_data = nwp_data_creator.make_dataset()
                self.files_manager.save_nwps(nwp_data)

    def create_image_dataset(self, parallel):
        if self.static_data['use_image']:
            image_data_creator = DatasetImageCreator(self.static_data, self.transformer, dates=self.dates,
                                                     is_online=self.is_online, parallel=parallel)
            image_data = self.files_manager.check_if_exists_image_data()
            if image_data is None:
                image_dates = image_data_creator.make_dataset()
                self.files_manager.save_images(image_dates)


    def concat_lagged_data(self, data, var_name, var_data, lstm_lags=None):
        if isinstance(data, pd.Series):
            data = data.to_frame()
        data = data.sort_index()
        data = fix_timeseries_dates(data, freq=self.static_data['ts_resolution'])
        dates = pd.date_range(data.index[0], data.index[-1],
                              freq=self.static_data['ts_resolution'])
        data_temp = pd.DataFrame(index=dates)
        if lstm_lags is not None:
            var_lags = []
            for l in lstm_lags:
                if isinstance(l, int) or isinstance(l, np.integer):
                    if l < 0 or l in var_data['lags']:
                        var_lags.append(l)
                else:
                    var_lags.append(l)
        else:
            var_lags =  var_data['lags']
        if len(var_lags) == 0:
            return None
        for lag in var_lags:
            if var_data['source'] == 'target':
                col = self.static_data['project_name']
                if col not in data.columns:
                    col = var_name
            elif var_data['source'] in {'nwp_dataset', 'index', 'grib'}:
                col = var_name
            elif var_data['source'].endswith('.csv'):
                if var_name in data.columns:
                    col = var_name
                else:
                    col = data.columns[0]
            else:
                col = var_data['source']
            if isinstance(lag, float):
                raise ValueError('Lag must be integer or string')
            freq = self.static_data['ts_resolution']
            if isinstance(lag, int) or isinstance(lag, np.integer):
                data_temp[f'{var_name}_lag_{lag}'] = data[col].shift(periods=-lag, freq=freq)
            elif isinstance(lag, str):
                lylags = pd.DataFrame()
                for d in data_temp.index:
                    try:
                        lags = last_year_lags(d, self.static_data['country'], freq=freq)
                        loads = data[col].loc[lags]
                        loads = pd.DataFrame(np.expand_dims(loads.values, axis=0), index=[d],
                                             columns=[f'{var_name}_lag_ly{i}' for i in range(len(lags))])
                        lylags = pd.concat([lylags, loads])
                    except:
                        continue
                data_temp = pd.concat([data_temp, lylags], axis=1)
        if var_data['use_diff_between_lags']:
            diff_df = []
            for lag1 in var_lags:
                for lag2 in var_lags:
                    if isinstance(lag1, str) or isinstance(lag2, str):
                        continue
                    if np.abs(lag1) > 3 or np.abs(lag2) > 3:
                        continue
                    if lag1 > lag2:
                        diff = data_temp[f'{var_name}_lag_{lag1}'] - data_temp[f'{var_name}_lag_{lag2}']
                        diff = diff.to_frame(f'Diff_{var_name}_lag{lag1}_lag{lag2}')
                        diff2 = np.square(diff)
                        diff2.columns = [f'Diff2_{var_name}_lag{lag1}_lag{lag2}']
                        diff_df.append(pd.concat([diff, diff2], axis=1))
            data_temp = pd.concat([data_temp] + diff_df, axis=1)

        return data_temp

    def create_autoregressive_dataset(self, lag_lstm=None):
        variables = dict([(var_data['name'], var_data) for var_data in self.static_data['variables']
                          if var_data['type'] == 'timeseries' and var_data['source'] == 'target'])
        if len(variables) == 0:
            return None
        data_arma = pd.DataFrame()
        for var_name, var_data in variables.items():
            data = self.data.copy(deep=True)

            data = self.concat_lagged_data(data, var_name, var_data, lstm_lags=lag_lstm)
            if data is None:
                continue
            data = data.dropna(axis='index', how='any')
            data_arma = pd.concat([data_arma, data], axis=1)
        return data_arma

    def create_calendar_dataset(self, lag_lstm=None):
        # sin_transformer = lambda x: np.sin(x / period * 2 * np.pi)
        variables_index = dict([(var_data['name'], var_data) for var_data in self.static_data['variables']
                                if var_data['type'] == 'calendar' and var_data['source'] == 'index'])

        if self.data is not None:
            data = self.data.copy(deep=True)
        else:
            data = None
        if not self.is_online:
            index = data.index
        else:
            if data is not None:
                max_lag = min([min(var_data['lags']) for var_data in self.static_data['variables']])
                index = pd.date_range(data.index[0] + pd.DateOffset(hours=max_lag),
                                      self.dates[-1] + pd.DateOffset(hours=47),
                                      freq=self.static_data['ts_resolution'])
            else:
                if self.static_data['horizon_type'] == 'intra-ahead':
                    index = pd.date_range(self.dates[0], self.dates[-1] + pd.DateOffset(hours=23), freq='h')
                else:
                    index = pd.date_range(self.dates[0], self.dates[-1] + pd.DateOffset(hours=47), freq='h')
        data_temp = pd.DataFrame(index=index)
        for var_name, var_data in variables_index.items():
            if var_name == 'hour':
                values = index.hour.values
                period = 24
            elif var_name == 'month':
                values = index.month.values
                period = 12
            elif var_name == 'dayweek':
                values = index.dayofweek.values
                period = 7
            elif var_name == 'dayofyear':
                values = index.dayofyear.values
                period = 1
            elif var_name == 'sp_index':
                values = [sp_index(d, country=self.static_data['country']) for d in index]
            else:
                raise ValueError(f'Unknown variable {var_name} for index and calendar')

            values = pd.DataFrame(values, columns=[var_name], index=index)
            if len(var_data['lags']) > 1:
                values = self.concat_lagged_data(values, var_name, var_data, lstm_lags=lag_lstm)
            if values is not None:
                data_temp = pd.concat([data_temp, values], axis=1)
        variables_astral = dict([(var_data['name'], var_data) for var_data in self.static_data['variables']
                                 if var_data['type'] == 'calendar' and var_data['source'] == 'astral'])
        if len(variables_astral) == 0 and len(variables_index) == 0:
            return None
        for var_name, var_data in variables_astral.items():
            solpos = pvlib.solarposition.get_solarposition(index, self.static_data['coord'][0]
                                                           , self.static_data['coord'][1])
            if var_name not in {'azimuth', 'zenith'}:
                raise ValueError(f'Unknown variable {var_name} for astral and calendar. '
                                 f'Accepted values are azimuth, zenith')
            data_temp = pd.concat([data_temp, solpos[var_name].to_frame()], axis=1)
        data_temp = data_temp.dropna(axis='index', how='any')
        return data_temp

    def create_nwp_ts_datasets(self, lag_lstm=None):
        if self.static_data['NWP'] is None:
            return None
        data_types = self.static_data['base_data_types']
        if self.static_data['type'] in {'load', 'FA'}:
            compress = 'load'
            variables_nwp = dict([(var_data['name'], var_data) for var_data in self.static_data['variables']
                                  if var_data['type'] == 'timeseries' and var_data['source'] != 'target'
                                  and var_data['source'] == 'nwp_dataset'])
        else:
            compress = 'minimal'
            variables_nwp = dict([(var_data['name'], var_data) for var_data in self.static_data['variables']
                                    if var_data['type'] == 'nwp' and var_data['source'] != 'target'
                                    and var_data['source'] == 'grib'])
        data_nwp, metadata = self.data_feeder.feed_inputs('row_nwp',
                                                          merge=data_types['merge'], compress=compress, get_all=True)
        if self.static_data['ts_resolution'] == 'D':
            data_nwp['row_nwp'] = data_nwp['row_nwp'].resample('D').mean()
        data_row_nwp = pd.DataFrame()
        for var_name, var_data in variables_nwp.items():
            var_names = [name for name in data_nwp['row_nwp'].keys() if f'{var_name}_0' in name] \
                if var_name not in data_nwp['row_nwp'].columns else [var_name]
            for name in var_names:
                data = self.concat_lagged_data(data_nwp['row_nwp'][name], name, var_data, lstm_lags=lag_lstm)
                if data is None:
                    continue
                data = data.dropna(axis='index')
                data_row_nwp = pd.concat([data_row_nwp, data], axis=1)
        data_row_nwp = data_row_nwp.dropna(axis='index')
        return data_row_nwp


    def create_extra_ts_datasets(self, lag_lstm=None):
        variables_extra = dict([(var_data['name'], var_data) for var_data in self.static_data['variables']
                                    if var_data['type'] == 'timeseries' and var_data['source'] != 'target'
                                    and var_data['source'] != 'nwp_dataset'])

        if len(variables_extra) == 0:
            return None
        data_extra = pd.DataFrame()
        for var_name, var_data in variables_extra.items():
            name = var_name
            if var_data['source'].endswith('.csv'):
                if os.path.exists(var_data['source']):
                    data = pd.read_csv(var_data['source'], index_col=0, header=0, parse_dates=True)
                    if name in data.columns:
                        data = data[name].to_frame()
                    if var_data['transformer'] == 'fillnan':
                        data = self.transformer.transform(data, var_name)
                else:
                    raise ImportError(f"{var_data['source']} does not exists")
            else:
                data = pd.read_csv(self.static_data['filename'], index_col=0, header=0, parse_dates=True)
                if var_data['source'] not in data.columns:
                    raise ValueError(f"{var_data['source']} does not exists in main file columns. "
                                     f"Filename is {self.static_data['filename']}")
                data = data[var_data['source']].to_frame()
            data = self.concat_lagged_data(data, name, var_data, lstm_lags=lag_lstm)
            if data is None:
                continue
            data = data.dropna(axis='index')
            data_extra = pd.concat([data_extra, data], axis=1)
        data_extra = data_extra.dropna(axis='index')
        return data_extra

    def create_row_datasets(self):
        data_row = self.files_manager.check_if_exists_row_data()
        if data_row is None:
            data_arma = self.create_autoregressive_dataset()
            cols_obs = list(data_arma.columns) if data_arma is not None else []
            data_extra = self.create_extra_ts_datasets()
            cols_obs += list(data_extra.columns) if data_extra is not None else []
            data_calendar = self.create_calendar_dataset()
            data_row_all = pd.DataFrame()
            for data in [data_arma, data_extra, data_calendar]:
                if data is not None:
                    data_row_all = pd.concat([data_row_all, data], axis=1)
            data_row_all = data_row_all.dropna(axis='index', how='any' if not self.is_online else 'all')
            data_row = {'row_obs': data_row_all[cols_obs] if len(cols_obs) > 0 else None,
                        'calendar': data_calendar}
            self.files_manager.save_row_data(data_row)


    def create_target(self):
        variable = self.static_data['target_variable']
        data = self.files_manager.check_if_exists_target()
        if data is None:
            data = self.data.copy(deep=True)
            var_col = variable["source"] if variable["source"] in data.columns else self.static_data['_id']
            if self.static_data['horizon_type'] == 'multi-output':
                freq = self.static_data['ts_resolution']
                for hor, col in zip(variable['lags'], variable['columns']):
                    if isinstance(hor, float):
                        raise ValueError('Lag must be integer or string')
                    data[col] = data[var_col].shift(periods=-hor, freq=freq)
                data = data[variable['columns']].dropna(axis='index')
            elif self.static_data['horizon_type'] == 'day-ahead' and self.static_data['type'] == 'FA':
                data[variable['columns'][0]] = data[var_col].shift(-1)
                data = data[variable['columns'][0]].dropna(axis='index').to_frame()
            else:
                data[variable['columns'][0]] = data[var_col]
                data = data[variable['columns'][0]].dropna(axis='index').to_frame()

            self.files_manager.save_target(data)

    def merge_rnn_variables(self, dataset, data_df, var_list, var_lags):
        time_merge_variables = self.static_data['time_merge_variables'] if (
                len(self.static_data['time_merge_variables']) > 0) else None
        if time_merge_variables is not None:
            for new_var, value in time_merge_variables.items():
                merge_flag = all([v in var_list for v in value])
                if merge_flag:
                    var_for_merge = pd.DataFrame.from_dict(
                        {var_data['name']: [max([v for v in var_data['lags'] if not isinstance(v, str)]),
                                            min([v for v in var_data['lags'] if not isinstance(v, str)]),
                                            any([isinstance(v, str) for v in var_data['lags']])]
                         for var_data in self.static_data['variables']
                         if var_data['name'] in value}, orient='index').sort_values(0)
                    data_merged = pd.DataFrame()
                    for var, lag_lim in var_for_merge.iterrows():
                        lag_lim = lag_lim.values
                        cols = []
                        cols_new = []
                        for lag in var_lags:
                            if isinstance(lag, int) or isinstance(lag, float):
                                if lag <= lag_lim[0] and lag >= lag_lim[1]:
                                    cols.append(f'{var}_lag_{lag}')
                                    cols_new.append(f'{new_var}_lag_{lag}')
                            else:
                                if lag_lim[2]:
                                    ly_cols = [col for col in data_df.columns if 'ly' in col.split('_lag_')[-1]
                                               and f'{var}_lag' in col]
                                    cols += ly_cols
                                    cols_new += [col.replace(var, new_var) for col in ly_cols]
                        data1 = dataset[var][cols]
                        data1.columns = cols_new
                        data_merged = pd.concat([data1, data_merged], axis=1)
                        del dataset[var]
                    dataset[new_var] = data_merged
                    var_list = [v for v in var_list if v not in value] + [new_var]
        return dataset, set(var_list)


    def get_temporal_data(self, dataset, lags, data_type):
        if data_type == 'autoregressive':
            data_df = self.create_autoregressive_dataset(lag_lstm=lags)
        elif data_type == 'calendar':
            data_df = self.create_calendar_dataset(lag_lstm=lags)
        elif data_type == 'nwp_data':
            data_df = self.create_nwp_ts_datasets(lag_lstm=lags)
        else:
            data_df = self.create_extra_ts_datasets(lag_lstm=lags)

        if data_df is not None:
            var_ts = set([col.split('_lag_')[0] for col in data_df.columns if 'Diff' not in col])
        else:
            var_ts = []

        for var in var_ts:
            cols = []
            for lag in lags:
                if isinstance(lag, int) or isinstance(lag, np.integer) or isinstance(lag, float):
                    cols.append(f'{var}_lag_{lag}')
                else:
                    ly_cols = [col for col in data_df.columns if 'ly' in col.split('_lag_')[-1] and f'{var}_lag' in col]
                    cols += ly_cols
            for col in cols:
                if col not in data_df.columns:
                    data_df = pd.concat([data_df, pd.DataFrame(0, index=data_df.index,
                                                                         columns=[col])], axis=1)
            dataset[var] = data_df[cols]

        dataset, var_ts = self.merge_rnn_variables(dataset, data_df, var_ts, lags)
        return dataset, var_ts

    def create_lstm_dataset(self):
        data_lstm = self.files_manager.check_if_exists_lstm_data()
        if data_lstm is None:
            metadata = dict()
            data = dict()
            data['future'] = dict()
            data['past'] = dict()

            metadata['groups'] = []
            if 'global_past_lags' not in self.static_data.keys():
                raise ValueError('Cannot find global past lags in static_data. Check input configuration')
            if 'global_future_lags' not in self.static_data.keys():
                raise ValueError('Cannot find global future lags in static_data. Check input configuration')

            if isinstance(self.static_data['global_past_lags'], int):
                past_lags = [-i for i in range(1, self.static_data['global_past_lags'])]
            else:
                past_lags = [i for i in self.static_data['global_past_lags']]

            if isinstance(self.static_data['global_future_lags'], int):
                future_lags = [i for i in range(self.static_data['global_future_lags'])]
            else:
                future_lags = [i for i in self.static_data['global_future_lags']]

            metadata['future_lags'] = future_lags
            metadata['past_lags'] = past_lags


            data['past'], vars_list = self.get_temporal_data(data['past'], past_lags, 'autoregressive')
            past_vars = vars_list
            data['future'], vars_list = self.get_temporal_data(data['future'], future_lags, 'autoregressive')
            future_vars = vars_list
            data['past'], vars_list = self.get_temporal_data(data['past'], past_lags, 'nwp_data')
            past_vars = past_vars.union(vars_list)
            data['future'], vars_list = self.get_temporal_data(data['future'], future_lags, 'nwp_data')
            future_vars = future_vars.union(vars_list)
            data['past'], vars_list = self.get_temporal_data(data['past'], past_lags, 'extra')
            past_vars = past_vars.union(vars_list)
            data['future'], vars_list = self.get_temporal_data(data['future'], future_lags, 'extra')
            future_vars = future_vars.union(vars_list)
            data['past'], vars_list = self.get_temporal_data(data['past'], past_lags, 'calendar')
            past_vars = past_vars.union(vars_list)
            data['future'], vars_list = self.get_temporal_data(data['future'], future_lags, 'calendar')
            future_vars = future_vars.union(vars_list)
            metadata['past_variables'] = past_vars.difference(future_vars)

            dates = pd.DatetimeIndex([])
            for key, value in data.items():
                for key1, value1 in value.items():
                    if dates.shape[0] == 0:
                        dates = value1.index
                    else:
                        dates = dates.intersection(value1.index)
            metadata['dates'] = dates
            self.files_manager.save_lstm_data(data, metadata)

