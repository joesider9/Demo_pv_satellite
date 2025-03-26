import os
import yaml
import pandas as pd
from eforecast.common_utils.nwp_utils import create_area
from eforecast.common_utils.dataset_utils import fix_timeseries_dates


def initializer(static_data, online=False, read_data=True):
    """
    Func responsible to configure static_data attributes and load time series.

    """
    static_data['path_owner'] = os.path.join(static_data['sys_folder'], static_data['project_owner'])
    if not os.path.exists(static_data['path_owner']):
        os.makedirs(static_data['path_owner'])

    static_data['path_group'] = os.path.join(static_data['path_owner'],
                                             f"{static_data['projects_group']}"
                                             f"_ver{static_data['version_group']}")
    if not os.path.exists(static_data['path_group']):
        os.makedirs(static_data['path_group'])

    static_data['path_group_type'] = os.path.join(static_data['path_group'], static_data['type'])
    if not os.path.exists(static_data['path_group_type']):
        os.makedirs(static_data['path_group_type'])

    static_data['path_group_nwp'] = os.path.join(static_data['path_group'], 'nwp')
    if not os.path.exists(static_data['path_group_nwp']):
        os.makedirs(static_data['path_group_nwp'])
    static_data['path_project'] = os.path.join(static_data['path_group_type'],
                                               static_data['project_name'],
                                               static_data['horizon_type'])
    if not os.path.exists(static_data['path_project']):
        os.makedirs(static_data['path_project'])
    static_data['path_model'] = os.path.join(static_data['path_project'],
                                             f"model_ver{static_data['version_model']}")
    if not os.path.exists(static_data['path_model']):
        os.makedirs(static_data['path_model'])

    static_data['path_logs'] = os.path.join(static_data['path_project'], 'logging')
    if not os.path.exists(static_data['path_logs']):
        os.makedirs(static_data['path_logs'])

    static_data['path_data'] = os.path.join(static_data['path_model'], 'DATA')
    if not os.path.exists(static_data['path_data']):
        os.makedirs(static_data['path_data'])

    if static_data['NWP'] is not None:
        for nwp in static_data['NWP']:
            if nwp is not None:
                area, coord = create_area(static_data['coord'], nwp['resolution'])
                static_data['coord'] = coord
                nwp['area'] = area
                if isinstance(area, dict):
                    for key, value in area.items():
                        if (value[0][0] < static_data['area_group'][0][0]) or \
                                (value[0][1] < static_data['area_group'][0][1]) or \
                                (value[1][0] > static_data['area_group'][1][0]) or \
                                (value[1][1] > static_data['area_group'][1][1]):
                            raise ValueError(f'Area {key}  is smaller than static_data area group')
                else:
                    if (area[0][0] < static_data['area_group'][0][0]) or \
                            (area[0][1] < static_data['area_group'][0][1]) or \
                            (area[1][0] > static_data['area_group'][1][0]) or \
                            (area[1][1] > static_data['area_group'][1][1]):
                        raise ValueError(' Area from coords is smaller than static_data area group')

    static_data['_id'] = static_data['project_name']

    print('Static data of all projects created')
    return static_data
