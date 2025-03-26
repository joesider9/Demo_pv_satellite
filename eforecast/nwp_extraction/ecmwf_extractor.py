import datetime
import email
import imaplib
import os
import shutil
import sys
import tarfile
import copy

import joblib
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
# from joblib import Parallel
# from joblib import delayed


if sys.platform == 'linux':
    import pygrib
else:
    import cfgrib


class DownLoader:

    def __init__(self, date=None, path_nwp=None):
        from credentials import Credentials
        from credentials import JsonFileBackend
        if sys.platform == 'linux':
            file_cred = '/home/smartrue/Dropbox/current_codes/PycharmProjects/ECMWF_download/filemail.json'
        else:
            if os.path.exists('D:/'):
                file_cred = 'D:/Dropbox/current_codes/PycharmProjects/ECMWF_download/filemail.json'
            else:
                file_cred = 'C:/Dropbox/current_codes/PycharmProjects/ECMWF_download/filemail.json'
        if not os.path.exists(file_cred):
            file_cred = './filemail.json'
            if not os.path.exists(file_cred):
                raise ImportError('Cannot import credentials')
        if date is None:
            self.date = pd.to_datetime(datetime.datetime.now().strftime('%d%m%y'), format='%d%m%y')
        else:
            self.date = date
        self.credobj = Credentials([JsonFileBackend(file_cred)])
        file_name = str(self.date.year) + '/SIDERT' + self.date.strftime('%m%d') + '00UTC.tgz'
        self.filename = os.path.join(path_nwp, file_name)
        self.subject = 'Real Time data ' + self.date.strftime('%Y-%m-%d') + ' 00UTC'

    def download(self):
        try:
            imapSession = imaplib.IMAP4_SSL('imap.gmail.com')
            typ, accountDetails = imapSession.login(self.credobj.load('cred1'), self.credobj.load('cred2'))
            if typ != 'OK':
                raise ConnectionError('cannot connect')

            imapSession.select("ECMWF")
            typ, data = imapSession.search(None, '(SUBJECT "' + self.subject + '")')
            if typ != 'OK':

                raise IOError('cannot read emails')

            # Iterating over all emails
            for msgId in data[0].split():
                typ, messageParts = imapSession.fetch(msgId, '(RFC822)')
                if typ != 'OK':
                    raise IOError('cannot read messages')

                emailBody = messageParts[0][1]
                mail = email.message_from_bytes(emailBody)
                if mail.get_content_maintype() != 'multipart':
                    return
                for part in mail.walk():
                    if part.get_content_maintype() != 'multipart' and part.get('Content-Disposition') is not None:
                        print(self.filename)
                        fp = open(self.filename, 'wb')
                        fp.write(part.get_payload(decode=True))
                        fp.close()
            imapSession.close()
            imapSession.logout()
        except:
            print('Not able to download all attachments.', self.subject)


class EcmwfExtractor:

    def __init__(self, static_data, id_nwp, dates=None, online=False):
        self.dates_ts = None
        self.static_data = static_data
        self.is_online = online
        self.nwp_resolution = static_data['NWP'][id_nwp]['resolution']
        self.path_nwp = static_data['NWP'][id_nwp]['path_nwp_source']
        if not os.path.exists(self.path_nwp):
            os.makedirs(self.path_nwp)
        self.path_group_nwp = static_data['path_group_nwp']
        self.area_group = static_data['area_group']
        self.n_jobs = static_data['n_jobs']
        self.exclude_dates = pd.DatetimeIndex([])

        self.lat1 = self.area_group[0][0]
        self.lat2 = self.area_group[1][0]
        self.lon1 = self.area_group[0][1]
        self.lon2 = self.area_group[1][1]
        self.dates = dates

    def extract_pygrib1(self, date_of_measurement, file_name):

        # We get 48 hours forecasts. For every date available take the next 47 hourly predictions.

        nwps = dict()
        dates = pd.date_range(start=date_of_measurement, end=date_of_measurement + pd.DateOffset(hours=48), freq='h')
        for dt in dates:
            nwps[dt.strftime('%d%m%y%H%M')] = dict()
        grb = pygrib.open(file_name)
        temp = []
        for i in range(1, grb.messages + 1):
            g = grb.message(i)
            if g.cfVarNameECMF == 'u100':
                var = 'Uwind'
            elif g.cfVarNameECMF == 'v100':
                var = 'Vwind'
            elif g.cfVarNameECMF == 't2m':
                var = 'Temperature'
            elif g.cfVarNameECMF == 'tcc':
                var = 'Cloud'
            elif g.cfVarNameECMF == 'ssrd':
                var = 'Flux'
            dt = dates[g.endStep].strftime('%d%m%y%H%M')
            data, lat, long = g.data()  # Each "message" corresponds to a specific line on Earth.
            if var == 'Flux':
                if len(temp) == 0:
                    temp.append(data)
                else:
                    t = copy.deepcopy(data)
                    data = data - temp[0]
                    temp[0] = copy.deepcopy(t)
            nwps[dt]['lat'] = lat
            nwps[dt]['long'] = long
            nwps[dt][var] = data
        grb.close()
        del grb
        for dt in nwps.keys():
            Uwind = nwps[dt]['Uwind']
            Vwind = nwps[dt]['Vwind']
            r2d = 45.0 / np.arctan(1.0)
            wspeed = np.sqrt(np.square(Uwind) + np.square(Vwind))
            wdir = np.arctan2(Uwind, Vwind) * r2d + 180
            nwps[dt]['WS'] = wspeed
            nwps[dt]['WD'] = wdir
        return nwps

    def extract_cfgrib1(self, file_name):
        nwps = dict()
        data = cfgrib.open_dataset(file_name)
        dates = pd.to_datetime(data.valid_time.data, format='%Y-%m-%d %H:%M:%S').strftime('%d%m%y%H%M')
        Uwind = data.u100.data
        Vwind = data.v100.data
        temperature = data.t2m.data
        cloud = data.tcc.data
        flux = data.ssrd.data
        lat = data.latitude.data
        long = data.longitude.data
        r2d = 45.0 / np.arctan(1.0)
        wspeed = np.sqrt(np.square(Uwind) + np.square(Vwind))
        wdir = np.arctan2(Uwind, Vwind) * r2d + 180
        for i, dt in enumerate(dates):
            nwps[dt] = dict()
            nwps[dt]['lat'] = lat
            nwps[dt]['long'] = long
            nwps[dt]['Uwind'] = Uwind[i]
            nwps[dt]['Vwind'] = Vwind[i]
            nwps[dt]['WS'] = wspeed[i]
            nwps[dt]['WD'] = wdir[i]
            nwps[dt]['Temperature'] = temperature[i]
            nwps[dt]['Cloud'] = cloud[i]
            if i == 0:
                temp = copy.deepcopy(flux[i])
            elif i > 0:
                temp1 = copy.deepcopy(flux[i])
                flux[i] = flux[i] - temp
                temp = copy.deepcopy(temp1)
            nwps[dt]['Flux'] = flux[i]

        return nwps

    def extract_pygrib2(self, date_of_measurement, file_name):
        path_extract = os.path.join(self.path_nwp, 'extract/' + date_of_measurement.strftime('%d%m%y'))
        if not os.path.exists(path_extract):
            os.makedirs(path_extract)
        tar = tarfile.open(file_name)
        tar.extractall(path_extract)
        tar.close()
        dates = pd.date_range(start=date_of_measurement, end=date_of_measurement + pd.DateOffset(hours=48), freq='h')
        nwps = dict()
        temp = []
        for j, dt in enumerate(dates):
            file = os.path.join(path_extract,
                                'E_H6S' + date_of_measurement.strftime('%m%d') + '0000' + dt.strftime('%m%d') + str(
                                    dt.hour).zfill(
                                    2) + '001')
            if not os.path.exists(file):
                file = os.path.join(path_extract, 'E_H6S' + date_of_measurement.strftime(
                    '%m%d') + '0000' + date_of_measurement.strftime('%m%d') + '00011')
                if not os.path.exists(file):
                    continue

            grb = pygrib.open(file)
            nwps[dt.strftime('%d%m%y%H%M')] = dict()
            for i in range(1, grb.messages + 1):
                g = grb.message(i)
                if g.cfVarNameECMF == 'u100':
                    var = 'Uwind'
                elif g.cfVarNameECMF == 'v100':
                    var = 'Vwind'
                elif g.cfVarNameECMF == 't2m':
                    var = 'Temperature'
                elif g.cfVarNameECMF == 'tcc':
                    var = 'Cloud'
                elif g.cfVarNameECMF == 'ssrd':
                    var = 'Flux'

                data, lat, long = g.data()
                if var == 'Flux':
                    if len(temp) == 0:
                        temp.append(data)
                    else:
                        t = copy.deepcopy(data)
                        data = data - temp[0]
                        temp[0] = copy.deepcopy(t)
                nwps[dt.strftime('%d%m%y%H%M')]['lat'] = lat
                nwps[dt.strftime('%d%m%y%H%M')]['long'] = long
                nwps[dt.strftime('%d%m%y%H%M')][var] = data
            grb.close()
            del grb
        for dt in nwps.keys():
            try:
                if 'Uwind' not in nwps[dt].keys():
                    continue
                Uwind = nwps[dt]['Uwind']
                Vwind = nwps[dt]['Vwind']
                r2d = 45.0 / np.arctan(1.0)
                wspeed = np.sqrt(np.square(Uwind) + np.square(Vwind))
                wdir = np.arctan2(Uwind, Vwind) * r2d + 180
                nwps[dt]['WS'] = wspeed
                nwps[dt]['WD'] = wdir
            except:
                continue
        return nwps

    def extract_cfgrib2(self, date_of_measurement, file_name):
        path_extract = os.path.join(self.path_nwp, 'extract/' + date_of_measurement.strftime('%d%m%y'))
        if not os.path.exists(path_extract):
            os.makedirs(path_extract)
        else:
            shutil.rmtree(path_extract)
            if not os.path.exists(path_extract):
                os.makedirs(path_extract)
        tar = tarfile.open(file_name)
        tar.extractall(path_extract)
        tar.close()
        dates = pd.date_range(start=date_of_measurement, end=date_of_measurement + pd.DateOffset(hours=48), freq='h')
        nwps = dict()
        temp = []
        for i, dt in enumerate(dates):
            file = os.path.join(path_extract,
                                'E_H6S' + date_of_measurement.strftime('%m%d') + '0000' + dt.strftime('%m%d') + str(
                                    dt.hour).zfill(
                                    2) + '001')
            if not os.path.exists(file):
                file = os.path.join(path_extract, 'E_H6S' + date_of_measurement.strftime(
                    '%m%d') + '0000' + date_of_measurement.strftime('%m%d') + '00011')
                if not os.path.exists(file):
                    continue
            try:
                data = cfgrib.open_dataset(file)
            except:
                continue
            try:
                Uwind = data.u100.data
                Vwind = data.v100.data
                temperature = data.t2m.data
                cloud = data.tcc.data
                flux = data.ssrd.data
            except:
                continue
            if len(temp) == 0:
                temp.append(flux)
            else:
                t = copy.deepcopy(flux)
                flux = flux - temp[0]
                temp[0] = copy.deepcopy(t)
            lat = data.latitude.data
            long = data.longitude.data
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
            nwp['Temperature'] = temperature
            nwp['Cloud'] = cloud
            nwp['Flux'] = flux
            nwps[dt.strftime('%d%m%y%H%M')] = nwp
        return nwps

    def extract_pygrib3(self, date_of_measurement, file_name):

        dates = pd.date_range(start=date_of_measurement, end=date_of_measurement + pd.DateOffset(hours=48), freq='h')
        nwps = dict()
        temp = []
        for j, dt in enumerate(dates):
            file = os.path.join(file_name,
                                'E_H6S' + date_of_measurement.strftime('%m%d') + '0000' + dt.strftime('%m%d') + str(
                                    dt.hour).zfill(
                                    2) + '001')
            if not os.path.exists(file):
                file = os.path.join(file_name, 'E_H6S' + date_of_measurement.strftime(
                    '%m%d') + '0000' + date_of_measurement.strftime('%m%d') + '00011')

                if not os.path.exists(file):
                    continue

            grb = pygrib.open(file)
            nwps[dt.strftime('%d%m%y%H%M')] = dict()
            for i in range(1, grb.messages + 1):
                g = grb.message(i)
                if g.cfVarNameECMF == 'u100':
                    var = 'Uwind'
                elif g.cfVarNameECMF == 'v100':
                    var = 'Vwind'
                elif g.cfVarNameECMF == 't2m':
                    var = 'Temperature'
                elif g.cfVarNameECMF == 'tcc':
                    var = 'Cloud'
                elif g.cfVarNameECMF == 'ssrd':
                    var = 'Flux'

                data, lat, long = g.data()
                if var == 'Flux':
                    if len(temp) == 0:
                        temp.append(data)
                    else:
                        t = copy.deepcopy(data)
                        data = data - temp[0]
                        temp[0] = copy.deepcopy(t)
                nwps[dt.strftime('%d%m%y%H%M')]['lat'] = lat
                nwps[dt.strftime('%d%m%y%H%M')]['long'] = long
                nwps[dt.strftime('%d%m%y%H%M')][var] = data
            grb.close()
            del grb
        for dt in nwps.keys():
            Uwind = nwps[dt]['Uwind']
            Vwind = nwps[dt]['Vwind']
            r2d = 45.0 / np.arctan(1.0)
            wspeed = np.sqrt(np.square(Uwind) + np.square(Vwind))
            wdir = np.arctan2(Uwind, Vwind) * r2d + 180
            nwps[dt.strftime('%d%m%y%H%M')]['WS'] = wspeed
            nwps[dt.strftime('%d%m%y%H%M')]['WD'] = wdir
        return nwps

    def extract_cfgrib3(self, date_of_measurement, file_name):

        dates = pd.date_range(start=date_of_measurement, end=date_of_measurement + pd.DateOffset(hours=48), freq='h')
        nwps = dict()
        temp1 = []
        for i, dt in enumerate(dates):
            file = os.path.join(file_name,
                                'H6S' + date_of_measurement.strftime('%m%d') + '0000' + dt.strftime('%m%d') + str(
                                    dt.hour).zfill(
                                    2) + '001')
            if not os.path.exists(file):
                file = os.path.join(file_name, 'H6S' + date_of_measurement.strftime(
                    '%m%d') + '0000' + date_of_measurement.strftime('%m%d') + '00011')

                if not os.path.exists(file):
                    continue

            data = cfgrib.open_dataset(file)
            Uwind = data.u100.data
            Vwind = data.v100.data
            temp = data.t2m.data
            cloud = data.tcc.data
            flux = data.ssrd.data
            if len(temp1) == 0:
                temp1.append(flux)
            else:
                t = copy.deepcopy(flux)
                flux = flux - temp1[0]
                temp1[0] = copy.deepcopy(t)
            lat = data.latitude.data
            long = data.longitude.data
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
            nwp['Cloud'] = cloud
            nwp['Flux'] = flux
            nwps[dt.strftime('%d%m%y%H%M')] = nwp

        return nwps

    def nwps_extract_for_train(self, t):
        if not os.path.exists(os.path.join(self.path_nwp, 'extract')):
            os.makedirs(os.path.join(self.path_nwp, 'extract'))
        if not os.path.exists(os.path.join(self.path_nwp, t.strftime('%Y'))):
            os.makedirs(os.path.join(self.path_nwp, t.strftime('%Y')))
        file_name1 = os.path.join(self.path_nwp, f"{t.strftime('%Y')}/Sider2_{t.strftime('%Y%m%d')}.grib")
        file_name2 = os.path.join(self.path_nwp, t.strftime('%Y') + '/SIDERT' + t.strftime('%m%d') + '00UTC.tgz')
        file_name3 = os.path.join(self.path_nwp, t.strftime('%Y') + '/H6S' + t.strftime('%m%d') + '0000/')
        nwps = dict()
        if os.path.exists(file_name1):
            nwps = self.extract_pygrib1(t, file_name1) if sys.platform == 'linux' else self.extract_cfgrib1(file_name1)
        elif os.path.exists(file_name3):
            nwps = self.extract_pygrib3(t, file_name3) if sys.platform == 'linux' else self.extract_cfgrib3(t,
                                                                                                            file_name3)
        else:
            if not os.path.exists(file_name2):
                download = DownLoader(date=t, path_nwp=self.path_nwp)
                download.download()
            if os.path.exists(file_name2):
                try:
                    nwps = self.extract_pygrib2(t, file_name2) if sys.platform == 'linux' \
                        else self.extract_cfgrib2(t, file_name2)
                except:
                    download = DownLoader(date=t, path_nwp=self.path_nwp)
                    download.download()
                    nwps = self.extract_pygrib2(t, file_name2) if sys.platform == 'linux' else self.extract_cfgrib2(t,
                                                                                                        file_name2)
        # print(nwps)
        print('Extracted date ', t.strftime('%d%m%y'))
        if nwps:
            joblib.dump(nwps, os.path.join(self.path_group_nwp, f"ecmwf_{t.strftime('%d%m%y')}.pickle"))
            return t, 'Done'
        else:
            return t, 'Empty'

    def grib2dict_for_train(self, dates):
        with ProcessPoolExecutor(max_workers=self.static_data['n_jobs']) as executor:
            results = executor.map(self.nwps_extract_for_train, dates)
            results = list(results)
        # results = Parallel(n_jobs=self.n_jobs)(delayed(self.nwps_extract_for_train)(t) for t in dates)
        for res in results:
            if res[1] != "Done":
                self.exclude_dates = self.exclude_dates.append(pd.DatetimeIndex([res[0]]))
            else:
                print('nwp extracted for', res[0])
        if len(self.exclude_dates) > 100:
            for failure in self.exclude_dates:
                print(f'Date {failure} failed to extracted due {failure}')
            raise ImportError('Too many dates lost for nwp extraction')

    def grib2dict_for_train_online(self, dates_ts):
        for t in dates_ts:
            res = self.nwps_extract_for_train(t)
            if res[1] != "Done":
                print(f'Cannot extract date {res[0]} due to {res[1]}')
            else:
                print('nwp extracted for', res[0])

    def extract_nwps(self):
        self.dates_ts = pd.DatetimeIndex(self.dates).round('D').unique()

        dates = []
        for dt in self.dates_ts:
            if not os.path.exists(os.path.join(self.path_group_nwp, f"ecmwf_{dt.strftime('%d%m%y')}.pickle")) \
                    and dt not in self.exclude_dates:
                dates.append(dt)
        if not self.is_online:
            self.grib2dict_for_train(pd.DatetimeIndex(dates))
        else:
            self.grib2dict_for_train_online(pd.DatetimeIndex(dates))

        print('Nwp pickle file created for all date')
