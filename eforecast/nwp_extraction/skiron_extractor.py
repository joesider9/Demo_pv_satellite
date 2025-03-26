import ftplib
import os

import joblib
import numpy as np
import pandas as pd

# from joblib import Parallel
# from joblib import delayed
from concurrent.futures import ProcessPoolExecutor
import sys

if sys.platform == 'linux':
    import pygrib
else:
    import cfgrib


class SkironExtractor:
    """
    Provides numerical weather predictions (nwp) with a horizon of 72 hours.

    """

    def __init__(self, static_data, id_nwp, dates=None, is_for_test=False, is_online=False):
        self.static_data = static_data
        self.dates = dates
        self.is_for_test = is_for_test
        self.is_online = is_online
        self.nwp_resolution = static_data['NWP'][id_nwp]['resolution']
        self.path_nwp = static_data['NWP'][id_nwp]['path_nwp_source']
        self.path_group_nwp = static_data['path_group_nwp']
        self.exclude_dates = pd.DatetimeIndex([])
        self.area = static_data['area_group']
        self.n_jobs = static_data['n_jobs']

    def skiron_download(self, dt):

        with ftplib.FTP('ftp.mg.uoa.gr') as ftp:
            try:
                ftp.login('mfstep', '!lam')
                ftp.set_pasv(True)

            except Exception:
                print('Error in connection to FTP')
            local_dir = self.path_nwp + dt.strftime('%Y') + '/' + dt.strftime('%d%m%y')
            if not os.path.exists(local_dir):
                os.makedirs(local_dir)
            try:
                for hor in range(76):
                    target_filename = '/forecasts/Skiron/daily/005X005/' + dt.strftime(
                        '%d%m%y') + '/MFSTEP005_00' + dt.strftime('%d%m%y') + '_' + str(hor).zfill(3) + '.grb'
                    print('Trying to download nwp file MFSTEP005_00' + dt.strftime('%d%m%y') + '_' + str(hor).zfill(3)
                          + '.grb')
                    local_filename = local_dir + '/MFSTEP005_00' + dt.strftime('%d%m%y') + '_' + str(hor).zfill(
                        3) + '.grb'
                    if not os.path.exists(local_filename):
                        with open(local_filename, 'w+b') as f:
                            res = ftp.retrbinary('RETR %s' % target_filename, f.write)
                            count = 0
                            while not res.startswith('226 Transfer complete') and count <= 4:
                                print('Downloaded of file {0} is not compile.'.format(target_filename))
                                os.remove(local_filename)
                                with open(local_filename, 'w+b') as f:
                                    res = ftp.retrbinary('RETR %s' % target_filename, f.write)
                                    print('Success to download nwp fileMFSTEP005_00' + dt.strftime('%d%m%y')
                                          + '_' + str(hor).zfill(3) + '.grb')
                                count += 1
                                print('Failed to download nwp fileMFSTEP005_00' + dt.strftime('%d%m%y') + '_'
                                      + str(hor).zfill(3) + '.grb')
            except:
                print('Error downloading  {0} '.format(local_filename))
            ftp.quit()

    def extract(self, grb, la1, la2, lo1, lo2):
        nwps = dict()

        try:
            g = grb.message(21) if self.nwp_resolution == 0.05 else grb.message(1)
        except:
            g = grb.message(1)
        u_wind, lat, long = g.data(lat1=la1, lat2=la2, lon1=lo1, lon2=lo2)

        try:
            g = grb.message(22) if self.nwp_resolution == 0.05 else grb.message(2)
        except:
            g = grb.message(2)
        v_wind = g.data(lat1=la1, lat2=la2, lon1=lo1, lon2=lo2)[0]

        r2d = 45.0 / np.arctan(1.0)
        w_speed = np.sqrt(np.square(u_wind) + np.square(v_wind))
        w_dir = np.arctan2(u_wind, v_wind) * r2d + 180

        nwps['lat'] = lat
        nwps['long'] = long
        nwps['Uwind'] = u_wind
        nwps['Vwind'] = v_wind
        nwps['WS'] = w_speed
        nwps['WD'] = w_dir

        g = grb.message(3)
        x = g.data(lat1=la1, lat2=la2, lon1=lo1, lon2=lo2)
        nwps['Temperature'] = x[0]

        g = grb.message(7)
        x = g.data(lat1=la1, lat2=la2, lon1=lo1, lon2=lo2)
        nwps['Precipitation'] = x[0]

        g = grb.message(5)
        x = g.data(lat1=la1, lat2=la2, lon1=lo1, lon2=lo2)
        nwps['Cloud'] = x[0]

        g = grb.message(8)
        x = g.data(lat1=la1, lat2=la2, lon1=lo1, lon2=lo2)
        nwps['Flux'] = x[0]
        del x  # Why are we calling del? Is it too memory intensive?
        return nwps

    def nwps_extract_for_train(self, t):
        nwps = dict()
        dates = pd.date_range(start=t, end=t + pd.DateOffset(hours=50), freq='h')
        hors = [int(hor) for hor in range(0, 51)]
        for hor, dt in zip(hors, dates):

            type_prefix = 'MFSTEP005_00' if self.nwp_resolution == 0.05 else "MFSTEP_IASA_00"
            file_name = f"{t.strftime('%Y')}/{t.strftime('%d%m%y')}/{type_prefix}{t.strftime('%d%m%y')}_{str(hor).zfill(3)}.grb"
            nwp_path = os.path.join(self.path_nwp, file_name)  # MFSTEP_IASA_00010117_000
            if os.path.exists(nwp_path):
                try:
                    grb = pygrib.open(nwp_path)
                    la1 = self.area[0][0]
                    la2 = self.area[1][0]
                    lo1 = self.area[0][1]
                    lo2 = self.area[1][1]

                    nwps[dt.strftime('%d%m%y%H%M')] = self.extract(grb, la1, la2, lo1, lo2)
                    grb.close()
                    del grb
                    print('nwps extracted from ', nwp_path)
                except Exception:
                    pass
        if nwps:
            joblib.dump(nwps, os.path.join(self.path_group_nwp, f"skiron_{t.strftime('%d%m%y')}.pickle"))
            return t, 'Done'
        else:
            return t, 'Empty'

    def grib2dict_for_train(self, dates):
        with ProcessPoolExecutor(max_workers=self.static_data['n_jobs']) as executor:
            results = executor.map(self.nwps_extract_for_train, dates)
            results = list(results)
        # results = Parallel(n_jobs=self.n_jobs, timeout=9999)(delayed(self.nwps_extract_for_train)(t) for t in dates)
        for res in results:
            if res[1] != "Done":
                self.exclude_dates = self.exclude_dates.append(pd.DatetimeIndex([res[0]]))
        if len(self.exclude_dates) > 20:
            for failure in self.exclude_dates:
                print(f'Date {failure[0]} failed to extracted due {failure[1]}')
            raise ImportError('Too many dates lost for nwp extraction')

    def grib2dict_for_train_cfgrib(self, dates):
        with ProcessPoolExecutor(max_workers=self.static_data['n_jobs']) as executor:
            results = executor.map(self.nwps_extract_for_train_cfgrib, dates)
            results = list(results)
        # results = Parallel(n_jobs=self.n_jobs, timeout=9999)(delayed(self.nwps_extract_for_train_cfgrib)(t) for t in dates)
        for res in results:
            if res[1] != "Done":
                self.exclude_dates = self.exclude_dates.append(pd.DatetimeIndex([res[0]]))
        if len(self.exclude_dates) > 20:
            for failure in self.exclude_dates:
                print(f'Date {failure[0]} failed to extracted due {failure[1]}')
            raise ImportError('Too many dates lost for nwp extraction')

    def extract_nwps(self):
        self.dates = self.dates.round('D').unique()
        dates = []
        for dt in self.dates:
            if not os.path.exists(os.path.join(self.path_group_nwp, f"skiron_{dt.strftime('%d%m%y')}.pickle")) \
                    and not dt in self.exclude_dates:
                dates.append(dt)
        dates_to_load = pd.DatetimeIndex(dates)
        if sys.platform == 'linux':
            if len(dates) > 0:
                self.grib2dict_for_train(dates_to_load)
        else:
            if len(dates) > 0:
                self.grib2dict_for_train_cfgrib(dates_to_load)

    def extract_cfgrib(self, data, la1, la2, lo1, lo2):
        nwp = dict()
        lat = data.latitude.data
        long = data.longitude.data
        lats = np.where((lat >= la1) & (lat <= la2))[0]
        longs = np.where((long >= lo1) & (long <= lo2))[0]

        Uwind = data.u10.data[lats, :][:, longs]
        Vwind = data.v10.data[lats, :][:, longs]
        temp = data.t.data[lats, :][:, longs]
        precip = data.tp.data[lats, :][:, longs]
        cloud = data.tcc.data[lats, :][:, longs]
        flux = data.unknown.data[lats, :][:, longs]
        lat = data.latitude.data[lats].reshape(-1, 1)
        long = data.longitude.data[longs].reshape(1, -1)
        r2d = 45.0 / np.arctan(1.0)
        wspeed = np.sqrt(np.square(Uwind) + np.square(Vwind))
        wdir = np.arctan2(Uwind, Vwind) * r2d + 180

        nwp = dict()
        nwp['lat'] = lat
        nwp['long'] = long
        nwp['Uwind'] = Uwind
        nwp['Vwind'] = Vwind
        nwp['WS'] = wspeed
        nwp['WD'] = wdir
        nwp['Temperature'] = temp
        nwp['Precipitation'] = precip
        nwp['Cloud'] = cloud
        nwp['Flux'] = flux
        return nwp

    def nwps_extract_for_train_cfgrib(self, t):
        nwps = dict()
        dates = pd.date_range(start=t, end=t + pd.DateOffset(hours=50), freq='h')
        hors = [int(hor) for hor in range(0, 51)]
        for hor, dt in zip(hors, dates):

            type_prefix = 'MFSTEP005_00' if self.nwp_resolution == 0.05 else "MFSTEP_IASA_00"
            file_name = f"{t.strftime('%Y')}/{t.strftime('%d%m%y')}/{type_prefix}{t.strftime('%d%m%y')}_{str(hor).zfill(3)}.grb"
            nwp_path = os.path.join(self.path_nwp, file_name)  # MFSTEP_IASA_00010117_000
            if os.path.exists(nwp_path):
                try:
                    data = cfgrib.open_dataset(nwp_path)
                    la1 = self.area[0][0]
                    la2 = self.area[1][0]
                    lo1 = self.area[0][1]
                    lo2 = self.area[1][1]

                    nwps[dt.strftime('%d%m%y%H%M')] = self.extract_cfgrib(data, la1, la2, lo1, lo2)
                    print('nwps extracted from ', nwp_path)
                except Exception:
                    pass
        if nwps:
            joblib.dump(nwps, os.path.join(self.path_group_nwp, f"skiron_{t.strftime('%d%m%y')}.pickle"))
            return t, 'Done'
        else:
            return t, 'Empty'
