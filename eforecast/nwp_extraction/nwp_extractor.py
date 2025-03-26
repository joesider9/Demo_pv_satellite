import os
import shutil
import sys
import joblib

from eforecast.common_utils.logger import create_logger
from eforecast.nwp_extraction.ecmwf_extractor import EcmwfExtractor
from eforecast.nwp_extraction.skiron_extractor import SkironExtractor
from eforecast.nwp_extraction.gfs_extractor import GfsExtractor


class NwpExtractor:
    def __init__(self, static_data, recreate=False, is_online=False, dates=None):
        self.recreate = recreate
        self.nwp_extractors = {}
        self.static_data = static_data
        self.nwp_models = static_data['NWP']
        try:
            self.load()
        except:
            pass
        self.dates = dates
        self.is_online = is_online
        self.path_group = self.static_data['path_group']
        if not os.path.exists(self.path_group):
            os.makedirs(self.path_group)
        self.path_group_nwp = self.static_data['path_group_nwp']

    def extract(self):
        if self.recreate:
            shutil.rmtree(self.path_group_nwp)
            os.makedirs(self.path_group_nwp)

        for id_nwp, nwp in enumerate(self.nwp_models):
            if nwp['model'] is not None:
                nwp_model = nwp['model']
                print(f'Start extracting nwps {nwp_model}')
                if nwp_model in self.nwp_extractors.keys():
                    nwp_extractor = self.nwp_extractors[nwp_model]
                    if not hasattr(nwp_extractor, 'is_online'):
                        setattr(nwp_extractor, 'is_online', False)
                else:
                    if nwp_model == 'skiron' and sys.platform == 'linux':
                        nwp_extractor = SkironExtractor(self.static_data, id_nwp, dates=self.dates)
                    elif nwp_model == 'ecmwf':
                        nwp_extractor = EcmwfExtractor(self.static_data, id_nwp, dates=self.dates, online=self.is_online)
                    elif nwp_model == 'gfs':
                        nwp_extractor = GfsExtractor(self.static_data, id_nwp, dates=self.dates, online=self.is_online)
                    elif nwp_model == 'openweather':
                        pass
                    else:
                        raise ValueError('Cannot recognize nwp model')
                nwp_extractor.extract_nwps()
                self.nwp_extractors[nwp_model] = nwp_extractor
        print('Finish extract nwps')
        self.save()
        print('NWPs extracted successfully')
        return 'Done'

    def load(self):
        if os.path.exists(os.path.join(self.path_group_nwp, 'nwp_extraction.pickle')):
            try:
                tmp_dict = joblib.load(os.path.join(self.path_group_nwp, 'nwp_extraction.pickle'))
                self.__dict__.update(tmp_dict)
            except:
                raise ImportError('Cannot open CNN model')
        else:
            raise ImportError('Cannot find CNN model')

    def save(self):
        tmp_dict = {}
        for k in self.__dict__.keys():
            if k not in ['logger', 'static_data']:
                tmp_dict[k] = self.__dict__[k]
        joblib.dump(tmp_dict,os.path.join(self.path_group_nwp, 'nwp_extraction.pickle'), compress=9)