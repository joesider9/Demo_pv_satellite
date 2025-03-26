import requests

import os
import datetime
import joblib
import pandas as pd

from credentials import Credentials, JsonFileBackend

from eforecast.common_utils.date_utils import convert_timezone_dates

file_cred = os.path.join(os.path.abspath(__file__), 'openweather_key.json')


class OpenWeatherDownloader:

    def __init__(self, path_nwp, date=None):
        self.path_nwp = path_nwp
        if date is None:
            self.date = pd.to_datetime(datetime.datetime.now().strftime('%d%m%y'), format='%d%m%y')
        else:
            self.date = date
        self.dates = pd.date_range(self.date, self.date + pd.DateOffset(hours=42), freq='h')
        self.dates = pd.DatetimeIndex(convert_timezone_dates(self.dates))
        self.credobj = Credentials([JsonFileBackend(file_cred)])
        self.lat1, self.long1 = 38.04, 24.35
        self.lat2, self.long2 = 38.06, 24.34
        fname = 'openweather_' + self.date.strftime('%d%m%y') + '.pickle'
        self.filename = os.path.join(path_nwp, fname)

    def download(self):
        self.key = self.credobj.load('cred2')
        url1 = f'https://api.openweathermap.org/data/2.5/onecall?lat={self.lat1}&lon={self.long1}&exclude=daily,minutely' \
               f',current,alerts&appid={self.key}'
        url2 = f'https://api.openweathermap.org/data/2.5/onecall?lat={self.lat2}&lon={self.long2}&exclude=daily,minutely' \
               f',current,alerts&units=metric&appid={self.key}'

        response1 = requests.get(url1)
        response2 = requests.get(url2)
        if response1.status_code == 200 and response2.status_code == 200:
            try:
                nwp1 = response1.json()["hourly"]
                nwp2 = response2.json()["hourly"]
            except:
                raise ValueError('Openweather nwps are not downloaded correctly')
        else:
            raise ConnectionError('Openweather is not respond')
        for nwp in nwp1:
            del nwp['weather']
        for nwp in nwp2:
            del nwp['weather']
        nwp1 = pd.DataFrame().from_dict(nwp1)
        nwp2 = pd.DataFrame().from_dict(nwp2)
        nwp1.dt = pd.to_datetime(nwp1.dt, unit='s')
        nwp2.dt = pd.to_datetime(nwp2.dt, unit='s')
        nwp1 = nwp1.set_index('dt')
        nwp2 = nwp2.set_index('dt')
        columns = ['wind_speed', 'wind_deg']

        nwps = dict()
        for date in nwp1.index:
            if date.strftime('%d%m%y%H%M') not in nwps.keys():
                nwps[date.strftime('%d%m%y%H%M')] = dict()
            nwps[date.strftime('%d%m%y%H%M')]['marmari1'] = nwp1.loc[date, columns].astype(float).to_dict()
            nwps[date.strftime('%d%m%y%H%M')]['marmari2'] = nwp2.loc[date, columns].astype(float).to_dict()

        joblib.dump(nwps, self.filename)
