import os
import importlib

import pandas as pd
import numpy as np

from datetime import datetime

from eforecast.init.initialize import initializer
from eforecast.nwp_extraction.nwp_extractor import NwpExtractor
from eforecast.datasets.nwp_data.download_eumetsat_2 import run_sat_download
from eforecast.common_utils.date_utils import convert_timezone_dates


from run_model_script import predict

project_name_list = ['bittakos', 'short_term_image']

def impute_missing_values(df):
    dates = pd.date_range(df.index[0], df.index[-1], freq='15min')
    df_new = pd.DataFrame(index=dates, columns=df.columns)
    df_new.loc[df.index] = df
    ind_nan = np.where(df_new.isna().any(axis=1))
    if np.size(ind_nan) > 0:
        ind_nan = ind_nan[0]
        dates_nan = df_new.index[ind_nan]
        for dt in dates_nan:
            dts = pd.date_range(dt - pd.DateOffset(minutes=15), dt - pd.DateOffset(hours=1), freq='15min')
            df_new.loc[dt] = df_new.loc[dts].mean(axis=0, skipna=True)
    return df_new


def find_dates(df, last_date):
    dates = pd.date_range(df.index[0], last_date + pd.DateOffset(hours=23), freq='15min')
    dates_found = pd.DatetimeIndex(dates.difference(df.index).date).unique()
    return dates_found


def nwp_extraction(static_data, dates):
    # max_lag = [min(var_data['lags']) for var_data in static_data['variables']]
    # if len(max_lag) > 0:
    #     max_lag = min(max_lag)
    #     dates = dates.sort_values()
    #     dates = pd.date_range(dates[0] + pd.DateOffset(hours=max_lag), dates[-1])

    nwp_extractor = NwpExtractor(static_data, recreate=False, is_online=True, dates=dates)
    nwp_extractor.extract()


if __name__ == '__main__':
    name = '.'.join(project_name_list)
    module = importlib.import_module(f'{name}.configuration.config')
    config = getattr(module, 'config')
    static_data = initializer(config(), online=True)
    date_h = pd.to_datetime(datetime.now().strftime('%d%m%y %H:%M'), format='%d%m%y %H:%M').round('15min')
    # date_h = pd.to_datetime('2025-01-15 14:00', format='%Y-%m-%d %H:%M').round('15min')
    if static_data['Docker'] and not ('smartrue' in os.getcwd() or 'joesider' in os.getcwd()):
        date_h = convert_timezone_dates(pd.DatetimeIndex([date_h]), timezone1='UTC', timezone2='CET')[0]
    else:
        date_h = convert_timezone_dates(pd.DatetimeIndex([date_h]), timezone1='Europe/Athens', timezone2='CET')[0]
    print(date_h)
    date = pd.to_datetime(date_h.strftime('%d%m%y'), format='%d%m%y')
    path_pred = os.path.join(static_data['path_group'], 'predictions')
    if not os.path.exists(path_pred):
        os.makedirs(path_pred)
    print(f'{name} image model start at {date_h}')
    dates = pd.DatetimeIndex([date - pd.DateOffset(days=1), date])
    nwp_extraction(static_data, dates)
    dates_sat = [date_h - pd.DateOffset(minutes=int(60 * s)) for s in np.arange(0.5, 4.5, 0.25)]
    dates_sat = convert_timezone_dates(pd.DatetimeIndex(dates_sat), timezone1='CET', timezone2='UTC')
    # run_sat_download(dates_sat, static_data)
    if len(dates) > 0:
        pred_day_ahead = predict(pd.DatetimeIndex([date_h]), project_name_list,
                                 ['global', 'CrossViVit_test_1'])
        pred_day_ahead.to_csv(os.path.join(path_pred, f"PPC_pv_{date.strftime('%d_%m_%Y')}.csv"))
    print(f'{name} image model predictions are saved successfully')
