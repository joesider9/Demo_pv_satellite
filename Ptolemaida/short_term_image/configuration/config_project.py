"""
Define the attributes of the project - names, installed capacity, coordinates e.t.c.
    - PROJECT_NAME: the name of the project model
    - PROJECT_GROUP: The name of the group in which the project belongs. It is critical because all projects in a group
     shares same nwp files
    - PROJECT_OWNER: To whom belongs the project e.g. EDA for Azores
    - HORIZON_TYPE: day-ahead or multi-output
    - HORIZON: int 0 for day-head
    - COORDINATES: the coordinates of the site (lat, long) or of the area (lat_min, long_min, lat_max, long_max),
                   Could be a dictionary if project is regional
    - AREA_GROUP: The coordinates of the group area. It defines the grid area of nwp files that extracted from original
                  grib files
    - TYPE: load, pv, wind or fa
    - RATED_POWER: the installed capacity or None for load projects
    - NWP_MODELS: List with the NWP models. Could be None, ecmwf, gfs, skiron or openweather
    - DOCKER: True if runtime environment is docker

AUTHOR: G.SIDERATOS

Date: September 2022
"""
import numpy as np
import pandas as pd
from Ptolemaida.short_term_image.configuration.config_utils import *

DOCKER = False
if not ('smartrue' in os.getcwd() or 'PycharmProjects' in os.getcwd()):
    DOCKER = True
PROJECT_NAME = 'Ptolemaida'
PROJECT_GROUP = 'PPC_sat'
PROJECT_OWNER = 'PPC'

HORIZON_TYPE = 'multi-output'  # day-ahead, multi-output, intra-ahead
HORIZON = 4  # If horizon is 0 horizon type is day-ahead

VERSION_MODEL = 0
VERSION_GROUP = 2
 # dict for regional or list for point site. 4 elements list regional
COORDINATES = [40.47, 21.8]
AREA_GROUP = [[38, 19], [43, 24]]
IMAGE_COORD = [43, 35, 18, 27.5]
USE_IMAGES = True
SITE_INDICES = [55, 75]
IMAGE_SIZE = [110, 150]

NWP_MODELS = ['ecmwf']

RATED_POWER = 200
TYPE = 'pv'


FILE_NAME = f"{find_pycharm_path()}/Ptolemaida/data/pv_ts_15min.csv"
EVALUATION_START_DATE = ['2024-03-01 00:00:00', '2024-04-10 00:00:00']  # '%Y-%m-%d %H:%M or None
LOCAL_TIME_ZONE = 'CET'  # 'Europe/Athens', 'UTC', 'CET'
SITE_TIME_ZONE = 'Europe/Athens'  # 'Europe/Athens', 'UTC', 'CET'
COUNTRY = 'Greece'
TS_RESOLUTION = '15min'  # 'H' for hourly or 'D' for daily or '15min'
TIME_OFFSET = 0 if TYPE != 'load' else 9000

IS_GLOBAL = False
IS_FUZZY = True
IS_PROBABILISTIC = False


def check_coordinates(NWP):
    for coord in np.array(AREA_GROUP).ravel():
        if NWP is not None:
            for nwp in NWP:
                if (np.round(coord / nwp['resolution']) - coord / nwp['resolution']) > 1e-6:
                    raise ValueError(f"Latitude Longitude in area group should be multiple of NWP resolution "
                                     f"{nwp['resolution']}, "
                                     f"but it is {np.array(AREA_GROUP).ravel()}")


def config_project():
    folders = config_folders(DOCKER)
    n_jobs = define_n_jobs()
    if NWP_MODELS is None:
        NWP = None
    elif len(NWP_MODELS) == 0:
        NWP = None
    else:
        NWP = [find_nwp_attrs(nwp_model, folders['nwp_folder']) for nwp_model in NWP_MODELS]
        check_coordinates(NWP)
    project = {'project_name': PROJECT_NAME,
               'project_owner': PROJECT_OWNER,
               'projects_group': PROJECT_GROUP,
               'version_model': VERSION_MODEL,
               'version_group': VERSION_GROUP,
               'horizon_type': HORIZON_TYPE,
               'horizon': 0 if HORIZON_TYPE in {'day-ahead', 'intra-ahead'} else HORIZON,
               'rated': RATED_POWER,
               'coord': COORDINATES,
               'area_group': AREA_GROUP,
               'use_image': USE_IMAGES,
               'image_coord': IMAGE_COORD,
               'image_size': IMAGE_SIZE,
               'site_indices': SITE_INDICES,
               'type': TYPE,
               'NWP': NWP,
               'filename': FILE_NAME,
               'Evaluation_start': EVALUATION_START_DATE,
               'local_timezone': LOCAL_TIME_ZONE,
               'site_timezone': SITE_TIME_ZONE,
               'country': COUNTRY,
               'ts_resolution': TS_RESOLUTION,
               'time_offset': pd.DateOffset(hours=TIME_OFFSET) if TS_RESOLUTION == 'H' else pd.DateOffset(
                                                                                                days=TIME_OFFSET),
               'is_Global': IS_GLOBAL,
               'is_Fuzzy': IS_FUZZY,
               'is_probabilistic': IS_PROBABILISTIC,
               'Docker': DOCKER,
               'regional': True if isinstance(COORDINATES, dict)
                                   or (isinstance(COORDINATES, list) and len(COORDINATES) == 4) else False,
               'n_gpus': n_jobs['n_gpus'],
               'n_jobs': n_jobs['n_jobs'],
               'intra_op': n_jobs['intra_op']
               }
    project.update(folders)
    return project
