import os

import joblib
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import numpy as np
import pandas as pd
import astral
from astral.sun import sun

from eforecast.common_utils.date_utils import convert_timezone_dates
from eforecast.common_utils.nwp_utils import check_empty_multiple_nwp
from eforecast.common_utils.nwp_utils import get_lats_longs_by_date


class DatasetNWPsCreator:
    def __init__(self, static_data, transformer, dates=None, is_online=False, parallel=True):
        self.static_data = static_data
        self.transformer = transformer
        self.is_online = is_online
        self.parallel = parallel
        ts_res = str.lower(static_data['ts_resolution'])
        ts_res = ts_res if ts_res != '15min' else 'h'
        ts_res = ts_res if static_data['horizon_type'] == 'multi-output' else 'd'
        if self.is_online:
            max_lag = [min(var_data['lags']) for var_data in static_data['variables']]
            if len(max_lag) > 0:
                max_lag = min(max_lag)
                if isinstance(dates, list):
                    dates = pd.DatetimeIndex(dates)
                dates = dates.sort_values()
                if static_data['horizon_type'] == 'multi-output':
                    self.dates = pd.date_range(dates[0].floor(ts_res) + pd.DateOffset(hours=max_lag),
                                               dates[-1].floor(ts_res), freq='h')
                else:
                    self.dates = pd.date_range(dates[0].floor(ts_res) + pd.DateOffset(days=max_lag),
                                           dates[-1].floor(ts_res))
            else:
                self.dates = dates
        else:
            self.dates = dates.round(ts_res).unique()

        self.path_group_nwp = static_data['path_group_nwp']
        self.nwp_models = static_data['NWP']
        self.area_group = static_data['area_group']
        self.areas = self.static_data['NWP'][0]['area']
        self.coord = static_data['coord']
        self.n_jobs = static_data['n_jobs']
        self.variables = dict([(var_data['name'], var_data) for var_data in static_data['variables']
                               if var_data['type'] == 'nwp'])
        print(f"Dataset NWP creation started for project {self.static_data['_id']}")

    def make_dataset(self):
        if not isinstance(self.static_data['horizon'], int):
            raise ValueError('horizon parameter of static_data for multiple output (batch mode) should be integer')
        if isinstance(self.areas, dict):
            areas = [area for area in self.areas.keys()]
        else:
            areas = [self.static_data['project_name']]
        if not self.parallel:
            nwp_daily = []
            for t in tqdm(self.dates):
                nwp_daily.append(self.stack_daily_nwps(t, areas))
        else:
            with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
                nwp_daily = executor.map(self.stack_daily_nwps, self.dates, [areas] * len(self.dates))
                nwp_daily = list(nwp_daily)

        x_3d = dict()
        for area in areas:
            x_3d[area] = dict()
            for var in self.variables.keys():
                x_3d[area][var] = dict()
                for nwp in self.nwp_models:
                    x_3d[area][var][nwp['model']] = dict()
                    data = []
                    dates = pd.DatetimeIndex([])
                    for res in nwp_daily:
                        if res[area][var] is not None:
                            if len(self.variables[var]['lags']) == res[area][var][nwp['model']]['data'].shape[1]:
                                data.append(res[area][var][nwp['model']]['data'])
                                dates = dates.append(res[area][var][nwp['model']]['dates'])
                    data = np.vstack(data)

                    if var in self.transformer.variables_index.keys():
                        self.transformer.fit(data, var, data_dates=dates)
                        data = self.transformer.transform(data, var, data_dates=dates)
                    x_3d[area][var][nwp['model']]['data'] = data
                    x_3d[area][var][nwp['model']]['dates'] = dates

        return x_3d

    def stack_daily_nwps(self, t, areas):
        nwp_data = self.read_nwp_pickle(t)
        if self.static_data['horizon_type'] in {'intra-ahead', 'multi-output'}:
            if self.is_online:
                if self.static_data['horizon_type'] == 'intra-ahead':
                    p_dates = pd.date_range(t + pd.DateOffset(hours=1), t + pd.DateOffset(hours=24), freq='h')
                else:
                    p_dates = pd.DatetimeIndex([t])
            else:
                if self.static_data['horizon_type'] == 'intra-ahead':
                    p_dates = pd.date_range(t + pd.DateOffset(hours=1), t + pd.DateOffset(hours=24), freq='h')
                else:
                    if self.static_data['ts_resolution'] == 'D':
                        p_dates = pd.date_range(t + pd.DateOffset(hours=13), t + pd.DateOffset(hours=36), freq='h')
                    else:
                        p_dates = pd.DatetimeIndex([t])
        elif self.static_data['horizon_type'] == 'two-day-ahead':
            p_dates = pd.date_range(t + pd.DateOffset(hours=48), t + pd.DateOffset(hours=71), freq='h')
        else:
            p_dates = pd.date_range(t + pd.DateOffset(hours=24), t + pd.DateOffset(hours=47), freq='h')

        x_3d = self.create_inp_area(t, p_dates, nwp_data, areas)

        return x_3d

    def create_sample(self, nwp, lats, longs):
        x0 = nwp[np.ix_(lats, longs)]
        return x0

    def create_inp_lag(self, date, variable, nwps, area=None):
        dates_nwp = [date.round('h') + pd.DateOffset(hours=int(lag))
                     for lag in variable['lags']]
        try:
            if self.static_data['local_timezone'] != 'UTC':
                dates_nwp = convert_timezone_dates(pd.DatetimeIndex(dates_nwp),
                                                   timezone1=self.static_data['local_timezone'],
                                                   timezone2='UTC')
        except:
            return None
        inp_lag = []
        for date_nwp in dates_nwp:
            nwp = self.select_nwp(date, date_nwp, nwps)
            if nwp is None:
                return None
            if check_empty_multiple_nwp(nwp, self.variables):
                if not isinstance(nwps['lats'], dict):
                    inp = self.create_sample(nwp[variable['name']], nwps['lats'], nwps['longs'])
                else:
                    inp = self.create_sample(nwp[variable['name']], nwps['lats'][area], nwps['longs'][area])
                inp_lag.append(inp)
            else:
                return None
        if len(inp_lag) == 0:
            return None
        inp_lag = np.stack(inp_lag)
        return inp_lag


    def create_inp_dates(self, p_dates, variable, nwps, area=None):
        inp_dates = []
        dates = pd.DatetimeIndex([])
        for date in p_dates:
            inp = self.create_inp_lag(date, variable, nwps, area=area)
            if len(inp.shape) > 1:
                inp_dates.append(inp)
                dates = dates.append(pd.DatetimeIndex([date]))
        inp_dates = np.stack(inp_dates)
        return inp_dates, dates

    def create_inp_nwp_provider(self, t, p_dates, variable, nwp_data, area=None):
        inp_nwp_provider = dict()
        for id_nwp, nwps in enumerate(nwp_data):
            if nwps is None:
                print(f"NWP data not found for date {t}")
                inp_nwp_provider = None
                break
            else:
                try:
                    inp_nwp_provider[nwps['nwp_provider']] = dict()
                    if nwps['nwp_provider'] in variable['nwp_provider']:
                        inp_nwp_provider[nwps['nwp_provider']]['data'], inp_nwp_provider[nwps['nwp_provider']][
                            'dates'] = \
                            self.create_inp_dates(p_dates, variable, nwps, area=area)
                except:
                    inp_nwp_provider = None
                    break
        return inp_nwp_provider

    def create_inp_variables(self, t, p_dates, nwp_data, area=None):
        inp_var = dict()
        for name, variable in self.variables.items():
            inp_var[name] = self.create_inp_nwp_provider(t, p_dates, variable, nwp_data, area=area)
        return inp_var

    def create_inp_area(self, t, p_dates, nwp_data, areas):
        inp_area = dict()
        for area in areas:
            if area != 'area_group':
                inp_area[area] = self.create_inp_variables(t, p_dates, nwp_data, area=area)
        return inp_area

    def read_nwp_pickle(self, t):
        nwp_data = []
        lats = None
        longs = None
        for nwp_provider in self.nwp_models:
            if self.static_data['horizon_type'] in {'intra-ahead', 'multi-output'}:
                file_name1 = os.path.join(self.path_group_nwp,
                                          nwp_provider['model'] + '_' + (t - pd.DateOffset(days=1)).strftime(
                                              '%d%m%y') + '.pickle')
                if os.path.exists(file_name1):
                    try:
                        nwps_prev = joblib.load(file_name1)
                        p_dates = pd.date_range(t + pd.DateOffset(hours=1), t + pd.DateOffset(hours=48),
                                                freq='h').strftime(
                            '%d%m%y%H%M')
                        lats, longs = get_lats_longs_by_date(nwps_prev, p_dates, self.area_group, nwp_provider['area'],
                                                             nwp_provider['resolution'])
                    except:
                        os.remove(file_name1)
                        print(file_name1)
                        raise ImportError('Restart the training process. Some nwp files should reconstruct')
                else:
                    nwps_prev = dict()
            else:
                nwps_prev = dict()
            file_name = os.path.join(self.path_group_nwp,
                                     nwp_provider['model'] + '_' + t.strftime('%d%m%y') + '.pickle')
            if os.path.exists(file_name):
                try:
                    nwps = joblib.load(file_name)
                except:
                    os.remove(file_name)
                    print(file_name)
                    raise ImportError('Restart the training process. Some nwp files should reconstruct')
                p_dates = pd.date_range(t + pd.DateOffset(hours=1), t + pd.DateOffset(hours=48), freq='h').strftime(
                    '%d%m%y%H%M')
                lats, longs = get_lats_longs_by_date(nwps, p_dates, self.area_group, nwp_provider['area'],
                                                     nwp_provider['resolution'])
            else:
                nwps = dict()
                print(f"NWP file not found for date {t}")

            nwp_data.append({'nwp_provider': nwp_provider['model'], 'data': nwps, 'data_prev': nwps_prev,
                             'lats': lats, 'longs': longs})
        return nwp_data

    def select_nwp(self, date, date_nwp, nwps):
        nwp = None
        if self.static_data['horizon'] > 0:
            if date.hour < 10:
                data_day = 'data_prev'
            else:
                data_day = 'data'
            if date_nwp.strftime('%d%m%y%H%M') in nwps[data_day].keys():
                nwp = nwps[data_day][date_nwp.strftime('%d%m%y%H%M')]
            else:
                data_day = 'data' if data_day == 'data_prev' else 'data_prev'
                if date_nwp.strftime('%d%m%y%H%M') in nwps[data_day].keys():
                    nwp = nwps[data_day][date_nwp.strftime('%d%m%y%H%M')]
                else:
                    pass
        else:
            data_day = 'data'
            if date_nwp.strftime('%d%m%y%H%M') in nwps[data_day].keys():
                nwp = nwps[data_day][date_nwp.strftime('%d%m%y%H%M')]
            else:
                nwp = nwps['data_prev'][date_nwp.strftime('%d%m%y%H%M')]
        return nwp
