import numpy as np
import pandas as pd
import os
import sys
if sys.platform == 'win32':
    os.environ['GDAL_data'] = "C:/Users/joesider/anaconda3/envs/tf2x_env/Library/share/gdal"
import rasterio
from datetime import datetime as dt
from datetime import timedelta

Meas1 = 'downward_shortwave_radiation_flux'
Meas2 = 'precipitable_water_entire_atmosphere'
Meas3 = 'relative_humidity_2m_above_ground'
Meas4 = 'specific_humidity_2m_above_ground'
Meas5 = 'temperature_2m_above_ground'
Meas6 = 'total_cloud_cover_entire_atmosphere'
Meas7 = 'total_precipitation_surface'
Meas8 = 'u_component_of_wind_10m_above_ground'
Meas9 = 'v_component_of_wind_10m_above_ground'

meas_list = [Meas1, Meas2, Meas3, Meas4, Meas5, Meas6, Meas7, Meas8, Meas9]

def get_validDateTime(folder_dir, f):
    folder = folder_dir.replace('\\', '/').split('/')[-1]
    date_utc = dt.strptime(folder, '%Y%m%d%H') + timedelta(hours=int(f))
    return date_utc.strftime('%Y-%m-%d %H:%M:%S')


def return_3darray_from_dictionary(dictionary):
    arr = np.zeros((len(dictionary.keys()), len(dictionary[list(dictionary.keys())[0]]),
                    len(dictionary[list(dictionary.keys())[0]][0])))
    for i, k in enumerate(meas_list):
        arr[i, :, :] = dictionary[k]
    return arr


def transform_data_into_ndarray(folder_dir):
    # folder_dir = f'NOAA/GFS0P25/2021052100'
    names = {'downward_shortwave_radiation_flux': 'Flux',
             'relative_humidity_2m_above_ground': 'Humid',
             'temperature_2m_above_ground': 'Temperature',
             'total_cloud_cover_entire_atmosphere': 'Cloud',
             'u_component_of_wind_10m_above_ground': 'Uwind',
             'v_component_of_wind_10m_above_ground': 'Vwind',
             'precipitable_water_entire_atmosphere': 'Precip',
             'total_precipitation_surface': 'Precip_total',
             'specific_humidity_2m_above_ground': 'Specific_Humid'
             }
    gfs_data = {}
    sorted_int_files = sorted(int(f) for f in os.listdir(folder_dir))
    sorted_files = [str(f) for f in sorted_int_files]
    for f in sorted_files:
        file_dir = os.path.join(folder_dir, f)
        list_dir = [f for f in os.listdir(file_dir) if not f.startswith('.')]
        c_temp = {}
        for i in range(len(list_dir)):

            with rasterio.open(os.path.join(file_dir, list_dir[i])) as data:
                temp = data.read(1)
            component = list_dir[i].split('.')[-2]
            date = list_dir[i].split('.')[-3].split('F')[0]
            hor = list_dir[i].split('.')[-3].split('F')[1]
            date_utc = dt.strptime(date, '%Y%m%d%H') + timedelta(hours=int(hor))
            date_utc.strftime('%Y-%m-%d %H:%M:%S')
            c_temp[names[component]] = temp
        gfs_data[pd.to_datetime(date_utc).strftime('%d%m%y%H%M')] = c_temp
    return gfs_data
