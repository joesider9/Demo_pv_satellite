import copy
import os
import cv2
import rasterio
import h5py
import joblib
from joblib import Parallel
from joblib import delayed
from tqdm import tqdm
import numpy as np
import pandas as pd
import astral
from astral.sun import sun
from einops import rearrange
from einops import repeat

from eforecast.common_utils.date_utils import convert_timezone_dates


class DatasetImageCreator:

    def __init__(self, static_data, transformer, dates=None, is_online=False, parallel=False):
        self.static_data = static_data
        self.transformer = transformer
        self.is_online = is_online
        self.parallel = parallel
        ts_res = str.lower(static_data['ts_resolution'])
        if self.is_online:
            max_lag = [min(var_data['lags']) for var_data in static_data['variables']
                              if var_data['type'] == 'image']
            if len(max_lag) > 0:
                max_lag = min(max_lag)
                if isinstance(dates, list):
                    dates = pd.DatetimeIndex(dates)
                dates = dates.sort_values()

                self.dates = pd.date_range(dates[0].floor(ts_res) + pd.DateOffset(hours=max_lag),
                                           dates[-1].floor(ts_res), freq='15min')
            else:
                self.dates = dates
        else:
            self.dates = dates.round(ts_res).unique()
        self.path_sat = static_data['sat_folder']
        self.n_jobs = static_data['n_jobs']
        self.variables = dict([(var_data['name'], var_data) for var_data in static_data['variables']
                               if var_data['type'] == 'image'])
        self.static_variables = dict([(var_data['name'], var_data) for var_data in static_data['variables']
                               if var_data['type'] == 'static_image'])
        variables_names = set([var_data['name'] for var_data in static_data['variables']
                          if var_data['type'] == 'image'])
        self.extra_variables = dict([(var_data['name'], var_data) for var_data in static_data['variables']
                               if var_data['type'] in variables_names])
        for var in self.variables.keys():
            if var in self.transformer.variables_index.keys():
                self.transformer.fit(np.array([]), var, data_dates=dates)
        for var in self.extra_variables.keys():
            if var in self.transformer.variables_index.keys():
                self.transformer.fit(np.array([]), var, data_dates=dates)
        print(f"Dataset NWP creation started for project {self.static_data['_id']}")

    def make_dataset(self):
        if not os.path.exists(os.path.join(self.path_sat, 'processed')):
            os.makedirs(os.path.join(self.path_sat, 'processed'))
        if not self.parallel:
            dates_temp = []
            for t in tqdm(self.dates):
                dates_temp.append(self.stack_sat(t))
        else:
            dates_temp = Parallel(n_jobs=8)(
                delayed(self.stack_sat)(t) for t in tqdm(self.dates))
        dates = pd.DatetimeIndex([dt for dt in dates_temp if dt is not None])
        return dates

    def stack_sat(self, t):
        if not os.path.exists(os.path.join(self.path_sat, 'processed',
                                           f'satellite_{t.strftime("%Y_%m_%d__%H_%M")}.pkl')) or self.is_online:
            res = self.stack_hourly_sat(t)

            task = 'success'
            x_3d = dict()
            for var in self.variables.keys():
                x_3d[var] = dict()
                if res[var] is None:
                    task = 'failed'
                    return None
                data = res[var]['data']
                if len(data) > 0:
                    x_3d[var] = data
                else:
                    task = 'failed'
                    return None
            if task == 'failed':
                return None

            joblib.dump(x_3d, os.path.join(self.path_sat, 'processed',
                                           f'satellite_{t.strftime("%Y_%m_%d__%H_%M")}.pkl'))
        return t

    def stack_hourly_sat(self, t):
        x_3d = self.create_inp_variables(t)
        return x_3d

    def create_inp_lag(self, date, variable):
        inp_var = dict()
        inp_var['dates'] = pd.DatetimeIndex([date])
        dates_sat = pd.DatetimeIndex([date])
        try:
            if self.static_data['local_timezone'] != 'UTC':
                dates_sat = convert_timezone_dates(dates_sat,
                                                   timezone1=self.static_data['local_timezone'],
                                                   timezone2='UTC')
        except:
            return None
        inp_lag = []
        for date_sat in dates_sat:
            sat = None
            start = date_sat + pd.DateOffset(minutes=5)
            end = date_sat
            while sat is None:
                end = end - pd.DateOffset(minutes=15)
                if (start - end).seconds // 3600 >= 13:
                    break
                sat = self.read_sat_h5(start, end, variable)
            if sat is None or np.all(sat <= 0):
                return None
            sat = self.transformer.transform(sat, variable['name'])
            if sat is None:
                return None
            inp_lag.append(np.expand_dims(sat, axis=0))
        if len(inp_lag) == 0:
            return None
        inp_lag = np.vstack(inp_lag)
        inp_lag = rearrange(inp_lag, 'l c w h -> l w h c')
        inp_var['data'] = np.expand_dims(inp_lag, axis=0)
        return inp_var

    def create_inp_variables(self, t):
        inp_var = dict()
        for name, variable in self.variables.items():
            inp_var[name] = self.create_inp_lag(t, variable)
        # if inp_var['Cloud_Mask'] is not None:
        #     pass
        return inp_var

    def read_sat_h5(self, start, end, variable):
        dates = pd.date_range(end, start, freq='h').ceil('H').sort_values(ascending=False)
        files = []
        dates_files = []
        for date in dates:
            path_file = os.path.join(self.path_sat, f'{date.year}_{date.strftime("%B")}_{date.day}', f'{date.hour}')
            if not os.path.exists(path_file):
                path_file = os.path.join(os.path.split(self.path_sat)[0],
                                         f'{date.year}_{date.strftime("%B")}_{date.day}', f'{date.hour}')
                if not os.path.exists(path_file):
                    continue
            if variable['name'] == 'Cloud_Mask':
                sat_abbr = 'CLOUD'
            elif variable['name'] in {'RBG', 'Infrared', 'Infrared1', 'Infrared2', 'target_RBG'}:
                sat_abbr = 'IR'
            else:
                raise ValueError('Unknown satellite variable name')
            for image_file in os.listdir(path_file):
                if str.upper(sat_abbr) in str.upper(image_file):
                    date_file = pd.to_datetime(image_file.split('.')[0].split('_')[-1], format='%Y%m%dT%H%M%SZ')
                    if end <= date_file <= start:
                        files.append(os.path.join(path_file, image_file))
                        dates_files.append(date_file)
        if len(files) == 0:
            return None
        else:
            dates_files, index = pd.DatetimeIndex(dates_files).sort_values(return_indexer=True, ascending=False)
            for i in index:
                try:
                    file = files[i]
                    data = h5py.File(file, 'r')
                    return rearrange([data[band][()].astype('float') for band in variable['bands']],
                                     'b w c -> b w c')
                except:
                    pass
        return None

    def read_sat_tiff(self, start, end, variable):
        variable_name = variable['name']
        dates = pd.date_range(end, start, freq='h').ceil('H').sort_values(ascending=False)
        files = []
        dates_files = []
        for date in dates:
            path_file = os.path.join(self.path_sat, f'{date.year}_{date.strftime("%B")}_{date.day}', f'{date.hour}')
            if not os.path.exists(path_file):
                path_file = os.path.join(os.path.split(self.path_sat)[0],
                                         f'{date.year}_{date.strftime("%B")}_{date.day}', f'{date.hour}')
                if not os.path.exists(path_file):
                    continue
            for image_file in os.listdir(path_file):
                if str.upper(variable_name.split('_')[0]) in str.upper(image_file):
                    date_file = pd.to_datetime(image_file.split('.')[0].split('_')[-1], format='%Y%m%dT%H%M00Z')
                    if end <= date_file <= start:
                        files.append(os.path.join(path_file, image_file))
                        dates_files.append(date_file)
        if len(files) == 0:
            return None
        else:
            dates_files, index = pd.DatetimeIndex(dates_files).sort_values(return_indexer=True, ascending=False)
            for i in index:
                try:
                    file = files[i]
                    with rasterio.open(file) as data:
                        img = np.array(data.read(1))
                        return img.astype('float')
                except:
                    pass
        return None
