import os
import shutil
import lmdb
import pickle

import numpy as np


def store_data_to_lmdb(gfs_data, day, path_nwp=None):

    lmdb_folder = os.path.join(path_nwp, 'Meteo_lmdb_folder')
    # try:
    num_hours = len(gfs_data)
    first_hour = list(gfs_data.keys())[0]
    map_size = num_hours * np.array([v for v in gfs_data[first_hour].values()]).nbytes * 1000

    if not os.path.exists(f'{lmdb_folder}/{day}'):
        os.makedirs(f'{lmdb_folder}/{day}')
    env = lmdb.open(f'{lmdb_folder}/{day}', map_size=map_size)
    with env.begin(write=True) as txn:
        for key, value in sorted(gfs_data.items()):
            key1 = f"{key}"
            txn.put(key1.encode("ascii"), pickle.dumps(np.array([v for v in gfs_data[key].values()])))
    env.close()
    # except Exception as e:
    #     pass
    #     g_logger.error(f'Could not right data to lmdb due to: {e}')


def delete_tiff_files(output_path=None):

    try:
        for filename in os.listdir(f'{output_path}'):
            file_path = os.path.join(f'{output_path}', filename)
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
    except Exception as e:
        error_log_message = f'Could not Delete Grib Files due to {e}'
        print(error_log_message)
        shutil.rmtree(file_path)


