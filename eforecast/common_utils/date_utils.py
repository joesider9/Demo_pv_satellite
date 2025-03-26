import numpy as np
import pandas as pd

from pytz import timezone
from workalendar.europe import Portugal
from workalendar.europe import Greece


def date_parser_fun(x):
    return pd.to_datetime(x, format='%Y%m%d %H:%M')


def convert_timezone(data, timezone1='Europe/Athens', timezone2='UTC'):
    def datetime_exists_in_tz(dt, tz):
        try:
            dt.tz_localize(tz)
            return True
        except:
            return False

    dates = data.index
    indices = [i for i, t in enumerate(dates) if datetime_exists_in_tz(t, tz=timezone(timezone1))]
    data = data.iloc[indices]
    dates = dates[indices]
    dates = dates.tz_localize(timezone(timezone1))
    dates = dates.tz_convert(timezone(timezone2))
    dates = [pd.to_datetime(dt.strftime('%d%m%y%H%M'), format='%d%m%y%H%M') for dt in dates]
    data.index = dates
    return data


def convert_timezone_dates(dates, timezone1='Europe/Athens', timezone2='UTC', return_indices=False):
    def datetime_exists_in_tz(dt, tz):
        try:
            dt.tz_localize(tz)
            return True
        except:
            return False

    indices = [i for i, t in enumerate(dates) if datetime_exists_in_tz(t, tz=timezone(timezone1))]
    dates = dates[indices]
    dates = dates.tz_localize(timezone(timezone1))
    dates = dates.tz_convert(timezone(timezone2))
    dates = [pd.to_datetime(dt.strftime('%d%m%y%H%M'), format='%d%m%y%H%M') for dt in dates]
    if return_indices:
        return dates, indices
    else:
        return dates


class Azores(Portugal):
    FIXED_HOLIDAYS = Portugal.FIXED_HOLIDAYS + (
        (4, 11, "Dia da Liberdade"),
        (7, 18, "Dia de Portugal"),
        (6, 29, " Dia de S. Pedro"),
    )

    def __init__(self):
        super(Portugal, self).__init__()
        self.include_epiphany = True
        self.include_all_saints = True
        self.include_boxing_day = True
        self.include_christmas_eve = True
        self.include_clean_monday = True
        self.include_easter_saturday = True
        self.include_easter_monday = True
        self.include_corpus_christi = True
        self.include_immaculate_conception = True
        self.variable_days = {'Clean Monday', 'Good Friday', 'Easter Saturday', 'Easter Sunday', 'Easter Monday'
            , 'Corpus Christi', 'Santo Cristo', 'Pombinha', 'Dia do Corpo de Deus'}

    def get_fixed_holidays(self, year):
        days = super().get_fixed_holidays(year)
        return days

    def get_variable_days(self, year):
        days = super().get_variable_days(year)
        if year > 2015 or year < 2013:
            days.append((self.get_easter_sunday(year) + pd.DateOffset(days=36), "Santo Cristo"))
            days.append((self.get_easter_sunday(year) + pd.DateOffset(days=50), "Pombinha"))
            days.append((self.get_easter_sunday(year) + pd.DateOffset(days=64), "Dia do Corpo de Deus"))
        return days

    def get_extras(self, year):
        days = []
        days.append(self.get_easter_sunday(year) + pd.DateOffset(days=36))
        days.append(self.get_easter_sunday(year) + pd.DateOffset(days=50))
        days.append(self.get_easter_sunday(year) + pd.DateOffset(days=64))
        return days


def sp_index(r, country='Greece'):
    if country == 'Greece':
        cal = Greece()
        cal.include_christmas_eve = True
        cal.include_easter_saturday = True
    elif country == 'Azores':
        cal = Azores()
        extra = cal.get_extras(r.year)
    else:
        raise NotImplementedError(f'Special index function not implemented for {country}')
    if cal.is_holiday(r):
        sp = 100
    else:
        if r.dayofweek == 6:
            sp = 50
        else:
            sp = 0
    return sp


def last_year_lags(r, country, freq='H'):
    if country == 'Greece':
        cal = Greece()
        cal.include_christmas_eve = True
        cal.include_easter_saturday = True
    else:
        raise NotImplementedError(f'Last year holidays index function not implemented for {country}')
    holidays = cal.get_calendar_holidays(r.year)
    holidays = pd.DataFrame(holidays, columns=['date', 'name'])
    holidays = holidays.set_index('date')
    holidays.index = pd.to_datetime(holidays.index)
    variable_days = ['Clean Monday', 'Good Friday', 'Easter Saturday', 'Easter Sunday', 'Easter Monday',
                     'Whit Monday', 'Pentecost']
    r1 = pd.to_datetime(r.strftime('%Y-%m-%d'))
    if cal.is_holiday(r):
        if r1 not in holidays.index:
            raise ValueError(f'date {r1} not in holidays index')
        name = holidays.loc[r1]
        if r1 in holidays.index and name.values[0] in variable_days:
            easter0 = cal.get_easter_sunday(r.year)
            easter1 = cal.get_easter_sunday(r.year - 1)

            if freq ==  'H':
                lag = int((easter0 - easter1) / pd.Timedelta(hours=1)) - 1
                lags = np.hstack([np.arange(lag, lag + 6), lag + 24, lag + 168])
                return [r - pd.DateOffset(hours=l) for l in lags]
            else:
                lag = int((easter0 - easter1) / pd.Timedelta(days=1)) - 1
                lags = np.hstack([lag, lag + 1, lag + 7])
                return [r - pd.DateOffset(days=l) for l in lags]
        else:
            if freq == 'D':
                return [r - pd.DateOffset(years=1) - pd.DateOffset(days=i) for i in [0, 1, 7]]
            else:
                return [r - pd.DateOffset(years=1) - pd.DateOffset(hours=i) for i in [0, 24, 168]]
    else:
        if freq == 'D':
            return [r - pd.DateOffset(years=1) - pd.DateOffset(days=i) for i in [0, 1, 7]]
        else:
            return [r - pd.DateOffset(years=1) - pd.DateOffset(hours=i) for i in [0, 24, 168]]
