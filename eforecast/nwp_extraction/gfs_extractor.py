import joblib
import numpy as np
import pandas as pd
# from joblib import Parallel, delayed
from concurrent.futures import ProcessPoolExecutor

import eforecast.nwp_extraction.lmdb_files.transformer as tf
import eforecast.nwp_extraction.lmdb_files.downloader as dl

from eforecast.nwp_extraction.lmdb_files.storer import *


class GfsExtractor:
    """
    Provides numerical weather predictions (nwp) with a horizon of 72 hours.

    """

    def __init__(self, static_data, id_nwp, dates=None, online=False):
        self.dates_ts = None
        self.is_online = online
        self.static_data = static_data
        self.nwp_resolution = static_data['NWP'][id_nwp]['resolution']
        self.path_nwp = static_data['NWP'][id_nwp]['path_nwp_source']
        self.path_group_nwp = static_data['path_group_nwp']
        self.area_group = static_data['area_group']
        self.n_jobs = static_data['n_jobs']
        self.exclude_dates = pd.DatetimeIndex([])

        self.lat1 = self.area_group[0][0]
        self.lat2 = self.area_group[1][0]
        self.lon1 = self.area_group[0][1]
        self.lon2 = self.area_group[1][1]

        self.dates = dates

    def nwps_extract_for_train(self, d):
        try:
            names = [[0, 'Flux'],
                     [2, 'Humid'],
                     [4, 'Temperature'],
                     [5, 'Cloud'],
                     [7, 'Uwind'],
                     [8, 'Vwind']
                     ]
            folder_dir = dl.store_files_in_gfs_filesystem(self.area_group, date=d, path_nwp=self.path_nwp,
                                                          use_parallel=self.is_online)
            gfs = tf.transform_data_into_ndarray(folder_dir)
            date = folder_dir.split('/')[-1]
            delete_tiff_files(output_path=folder_dir)

            for date, nwp in gfs.items():
                nwp['lat'] = np.arange(self.lat1, self.lat2 + self.nwp_resolution / 2,
                                       self.nwp_resolution).reshape(-1, 1)[::-1]
                nwp['long'] = np.arange(self.lon1, self.lon2 + self.nwp_resolution / 2,
                                        self.nwp_resolution).reshape(-1, 1).T

                if 'Uwind' in nwp.keys() and 'Vwind' in nwp.keys():
                    if nwp['Uwind'].shape[0] > 0 and nwp['Vwind'].shape[0] > 0:
                        r2d = 45.0 / np.arctan(1.0)
                        wspeed = np.sqrt(np.square(nwp['Uwind']) + np.square(nwp['Vwind']))
                        wdir = np.arctan2(nwp['Uwind'], nwp['Vwind']) * r2d + 180
                        nwp['WS'] = wspeed
                        nwp['WD'] = wdir
            print('Write...' + d.strftime('%Y%m%d'))
            joblib.dump(gfs, os.path.join(self.path_group_nwp, 'gfs_' + d.strftime('%d%m%y') + '.pickle'))
        except Exception as e:
            return d, e
        return d, 'Done'

    def grib2dict_for_train(self, dates):
        # results = self.nwps_extract_for_train(dates[0])
        with ProcessPoolExecutor(max_workers=self.static_data['n_jobs']) as executor:
            results = executor.map(self.nwps_extract_for_train, dates)
            results = list(results)
        # results = Parallel(n_jobs=self.n_jobs, timeout=9999)(delayed(self.nwps_extract_for_train)(t) for t in dates)
        for res in results:
            if res[1] != "Done":
                self.exclude_dates = self.exclude_dates.append(pd.DatetimeIndex([res[0]]))
            else:
                print('nwp extracted for', res[0])
        if len(self.exclude_dates) > 20:
            for failure in self.exclude_dates:
                print(f'Date {failure} failed to extracted')
            raise ImportError('Too many dates lost for nwp extraction')

    def grib2dict_for_train_online(self, dates_ts):
        for t in dates_ts:
            res = self.nwps_extract_for_train(t)
            if res[1] != "Done":
                raise ImportError(f'Cannot extract date {res[0]} due to {res[1]}')
            else:
                print('nwp extracted for', res[0])

    def extract_nwps(self):
        self.dates_ts = pd.DatetimeIndex(self.dates).round('D').unique()

        dates = []
        for dt in self.dates_ts:
            if not os.path.exists(os.path.join(self.path_group_nwp, f"gfs_{dt.strftime('%d%m%y')}.pickle")) \
                    and dt not in self.exclude_dates:
                dates.append(dt)
        if not self.is_online:
            self.grib2dict_for_train(pd.DatetimeIndex(dates))
        else:
            self.grib2dict_for_train_online(pd.DatetimeIndex(dates))
        print('Nwp pickle file created for all date')
