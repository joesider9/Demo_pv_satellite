import copy
import os
import joblib
import astral

# from joblib import Parallel
# from joblib import delayed
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Pool

from astral.sun import sun
import numpy as np
import pandas as pd

from eforecast.common_utils.dataset_utils import get_slice


class FilesManager:
    def __init__(self, static_data, is_online=False, train=True):
        self.static_data = static_data
        self.is_online = is_online
        self.train = train
        self.path_data = self.static_data['path_data']

    def file_target(self):
        if self.is_online:
            raise ValueError('Cannot create target files online')
        else:
            dataset_file = 'dataset_target_data.csv'
        return os.path.join(self.path_data, dataset_file)

    def check_if_exists_target(self):
        file = self.file_target()
        if not os.path.exists(file):
            return None
        else:
            data = pd.read_csv(file, index_col=0, header=0, parse_dates=True)
            if not self.is_online:
                data, _ = self.split(data, data.index)
            return data

    def save_target(self, row_data):
        file = self.file_target()
        row_data.to_csv(file)

    def remove_target_files(self):
        file = self.file_target()
        if os.path.exists(file):
            os.remove(file)

    def file_row_data(self):
        if self.is_online:
            dataset_file = 'dataset_row_data_online.pickle'
        else:
            dataset_file = 'dataset_row_data.pickle'
        return os.path.join(self.path_data, dataset_file)

    def check_if_exists_row_data(self):
        file = self.file_row_data()
        if not os.path.exists(file):
            if os.path.exists(file.replace('pickle', 'csv')):
                file = file.replace('pickle', 'csv')
        if not os.path.exists(file):
            return None
        else:
            data = joblib.load(file)
            if not self.is_online:
                for key, value in data.items():
                    if value is not None:
                        data[key], _ = self.split(value, value.index)
                    else:
                        data[key] = None
            else:
                for key, value in data.items():
                    if value is not None:
                        data[key], _ = self.remove_night_hours(value, value.index)
                    else:
                        data[key] = None
            dates = None
            for key, value in data.items():
                if value is not None:
                    dates = dates.intersection(value.index) if dates is not None else value.index
            for key, value in data.items():
                if value is not None:
                    data[key] = value.loc[dates]
                else:
                    data[key] = None
            return data

    def save_row_data(self, row_data):
        file = self.file_row_data()
        joblib.dump(row_data, file)

    def remove_row_data_files(self):
        file = self.file_row_data()
        if os.path.exists(file):
            os.remove(file)

    def file_lstm_data(self):
        if self.is_online:
            dataset_file = 'dataset_lstm_data_online.pickle'
        else:
            dataset_file = 'dataset_lstm_data.pickle'
        return os.path.join(self.path_data, dataset_file)

    def check_if_exists_lstm_data(self):
        file = self.file_lstm_data()
        if not os.path.exists(file):
            return None
        else:
            lstm_data_dict = joblib.load(file)
            lstm_data, metadata = lstm_data_dict['data'], lstm_data_dict['metadata']
            dates = pd.DatetimeIndex([])
            for key, value in lstm_data.items():
                for key1, value1 in value.items():
                    if dates.shape[0] == 0:
                        dates = value1.index
                    else:
                        dates= dates.intersection(value1.index)
            lstm_data_new = dict()
            for key, value in lstm_data.items():
                lstm_data_new[key] = dict()
                for key1, value1 in value.items():
                    lstm_data_new[key][key1] = value1.loc[dates]
            lstm_data = copy.deepcopy(lstm_data_new)
            metadata['dates'] = dates
            if not self.is_online:
                lstm_data, dates = self.split(lstm_data, dates)
                metadata['dates'] = dates
            else:
                lstm_data, dates = self.remove_night_hours(lstm_data, dates)
                metadata['dates'] = dates
            return {'data': lstm_data, 'metadata': metadata}

    def save_lstm_data(self, lstm_data, metadata):
        file = self.file_lstm_data()
        joblib.dump({'data': lstm_data, 'metadata': metadata}, file)

    def remove_lstm_data_files(self):
        file = self.file_lstm_data()
        if os.path.exists(file):
            os.remove(file)


    def file_nwps(self):
        if self.is_online:
            dataset_file = 'dataset_nwps_online.pickle'
        else:
           dataset_file = 'dataset_nwps.pickle'
        return os.path.join(self.path_data, dataset_file)

    def check_if_exists_nwp_data(self, get_all=False):
        file = self.file_nwps()
        if not os.path.exists(file):
            return None
        else:
            nwp_data = joblib.load(file)
            dates = pd.DatetimeIndex([])
            for area, area_data in nwp_data.items():
                for variable, var_data in area_data.items():
                    for vendor, nwp_provide_data in var_data.items():
                        if dates.shape[0] == 0:
                            dates = nwp_provide_data['dates']
                        else:
                            dates = dates.intersection(nwp_provide_data['dates'])
            data_new = dict()
            for area, area_data in nwp_data.items():
                data_new[area] = dict()
                for variable, var_data in area_data.items():
                    data_new[area][variable] = dict()
                    for vendor, nwp_provide_data in var_data.items():
                        data_new[area][variable][vendor] = dict()
                        ind = nwp_provide_data['dates'].get_indexer(dates)
                        data_new[area][variable][vendor]['dates'] = nwp_provide_data['dates'][ind]
                        data_new[area][variable][vendor]['data'] = nwp_provide_data['data'][ind]
            nwp_data = copy.deepcopy(data_new)
            if not self.is_online and not get_all:
                nwp_data, dates = self.split(nwp_data, dates)
            else:
                nwp_data, dates = self.remove_night_hours(nwp_data, dates)
            return nwp_data

    def save_nwps(self, nwp_data):
        file = self.file_nwps()
        joblib.dump(nwp_data, file)

    def remove_nwps(self):
        file = self.file_nwps()
        if os.path.exists(file):
            os.remove(file)

    def file_images(self):
        if self.is_online:
            dataset_file = 'dataset_images_online.pickle'
        else:
            dataset_file = 'dataset_images.pickle'
        return os.path.join(self.path_data, dataset_file)

    def check_if_exists_image_data(self):
        file = self.file_images()
        if not os.path.exists(file):
            return None
        else:
            data = joblib.load(file)
            if not self.is_online:
                if not isinstance(data, dict):
                    dates = data
                    data, dates = self.split(data, dates)
            else:
                if not isinstance(data, dict):
                    dates = data
                    data, dates = self.remove_night_hours(data, dates)
            return data

    def save_images(self, image_data):
        file = self.file_images()
        joblib.dump(image_data, file)

    def remove_images(self):
        file = self.file_images()
        if os.path.exists(file):
            os.remove(file)

    def daylight(self, date):
        try:
            l = astral.LocationInfo('Custom Name', 'My Region', self.static_data['local_timezone'],
                                    self.static_data['coord'][0], self.static_data['coord'][1])
            sun_attr = sun(l.observer, date=date, tzinfo=self.static_data['local_timezone'])
            sunrise = pd.to_datetime(sun_attr['dawn'].strftime('%Y%m%d %H:%M'), format='%Y%m%d %H:%M')
            sunset = pd.to_datetime(sun_attr['dusk'].strftime('%Y%m%d %H:%M'), format='%Y%m%d %H:%M')
            if sunrise <= date <= sunset + pd.DateOffset(hours=1):
                return date
            else:
                return None
        except:
            return None

    def remove_night_hours(self, data, dates):
        if self.static_data['remove_night_hours']:
            try:
                with Pool(self.static_data['n_jobs']) as pool:
                    daylight_dates = pool.map(self.daylight, dates)
                # with ProcessPoolExecutor(max_workers=self.static_data['n_jobs']) as executor:
                #     daylight_dates = executor.map(self.daylight, dates)
                #     daylight_dates = list(daylight_dates)

                # daylight_dates = Parallel(n_jobs=self.static_data['n_jobs'])(delayed(self.daylight)(date)
                #                                                                            for date in dates)
            except:
                daylight_dates = [self.daylight(date) for date in dates]
            daylight_dates = [d for d in daylight_dates if d is not None]
            dates_new = pd.DatetimeIndex(daylight_dates)
            ind_new = dates.get_indexer(dates_new)
            data = get_slice(data, ind_new)
            dates = dates_new
        return data, dates

    def split(self, data, dates):
        if self.static_data['remove_night_hours']:
            try:
                with Pool(self.static_data['n_jobs']) as pool:
                    daylight_dates = pool.map(self.daylight, dates)
                # with ProcessPoolExecutor(max_workers=self.static_data['n_jobs']) as executor:
                #     daylight_dates = executor.map(self.daylight, dates)
                #     daylight_dates = list(daylight_dates)
                # daylight_dates = Parallel(n_jobs=self.static_data['n_jobs'])(delayed(self.daylight)(date)
                #                                                                            for date in dates)
            except:
                daylight_dates = [self.daylight(date) for date in dates]
            daylight_dates = [d for d in daylight_dates if d is not None]
            dates_new = pd.DatetimeIndex(daylight_dates)
            ind_new = dates.get_indexer(dates_new)
            data = get_slice(data, ind_new)
            dates = dates_new
        if self.train:
            if not isinstance(self.static_data['Evaluation_start'], list):
                ind = np.where(dates <= self.static_data['Evaluation_start'])[0]
            else:
                if len(self.static_data['Evaluation_start']) == 2:
                    dates_eval = pd.date_range(self.static_data['Evaluation_start'][0],
                                               self.static_data['Evaluation_start'][0],
                                               freq=self.static_data['ts_resolution'])
                    dates_eval = dates_eval.intersection(dates)
                    dates_eval = dates.difference(dates_eval)
                    ind = dates.get_indexer(dates_eval)
                else:
                    dates_eval = pd.DatetimeIndex([])
                    for date in self.static_data['Evaluation_start']:
                        dates_eval = dates.append(pd.date_range(date, date + pd.DateOffset(hours=23),
                                                                freq=self.static_data['ts_resolution']))
                    dates_eval = dates_eval.intersection(dates)
                    dates_eval = dates.difference(dates_eval)
                    ind = dates.get_indexer(dates_eval)
        else:
            if not isinstance(self.static_data['Evaluation_start'], list):
                ind = np.where(dates > self.static_data['Evaluation_start'])[0]
            else:
                if len(self.static_data['Evaluation_start']) == 2:
                    dates_eval = pd.date_range(self.static_data['Evaluation_start'][0],
                                               self.static_data['Evaluation_start'][1],
                                               freq=self.static_data['ts_resolution'])
                    dates_eval = dates_eval.intersection(dates)
                    ind = dates.get_indexer(dates_eval)
                else:
                    dates_eval = pd.DatetimeIndex([])
                    for date in self.static_data['Evaluation_start']:
                        dates_eval = dates.append(pd.date_range(date, date + pd.DateOffset(hours=23),
                                                                freq=self.static_data['ts_resolution']))
                    dates_eval = dates_eval.intersection(dates)
                    ind = dates.get_indexer(dates_eval)

        return get_slice(data, ind), dates[ind]

    def file_cv_data(self, fuzzy=False):
        dataset_file = 'cv_mask.pickle' if not fuzzy else 'cv_mask_fuzzy.pickle'
        return os.path.join(self.path_data, dataset_file)

    def check_if_exists_cv_data(self, fuzzy=False):
        file = self.file_cv_data(fuzzy=fuzzy)
        if not os.path.exists(file):
            return None
        else:
            data = joblib.load(file)
            return data

    def save_cv_data(self, cv_mask, fuzzy=False):
        file = self.file_cv_data(fuzzy=fuzzy)
        joblib.dump(cv_mask, file)

    def remove_cv_data_files(self, fuzzy=False):
        file = self.file_cv_data(fuzzy=fuzzy)
        if os.path.exists(file):
            os.remove(file)
