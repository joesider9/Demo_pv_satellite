import os
import joblib

import numpy as np
import pandas as pd

from eforecast.datasets.files_manager import FilesManager


class FeatureSelector:
    def __init__(self, static_data, recreate=False, online=False, train=False):
        self.feature_selectors = dict()
        self.online = online
        self.train = train
        self.static_data = static_data
        self.is_Fuzzy = self.static_data['is_Fuzzy']
        self.clusters = dict()
        try:
            self.clusters = joblib.load(os.path.join(static_data['path_model'], 'clusters.pickle'))
        except:
            pass
        cluster_path = os.path.join(static_data['path_model'], 'global')
        self.clusters.update({'global': cluster_path})
        self.calendar_variables = [var_data['name'] for var_data in self.static_data['variables']
                                   if var_data['type'] == 'calendar']
        self.files_manager = FilesManager(static_data, is_online=online)

    def transform(self, selector_tag, cluster_path, data, transform_calendar=False):
        filename = os.path.join(cluster_path, 'feature_selectors.pickle')
        if os.path.exists(filename):
            self.feature_selectors.update(joblib.load(filename))
        if 'lstm' in selector_tag:
            feature_selector = self.feature_selectors[selector_tag]
            lag_ids = feature_selector['lags']
            var_names = feature_selector['variables']
            x_new = dict()
            past_shape = data['past'][list(data['past'].keys())[0]].shape[1]
            future_shape = data['future'][list(data['future'].keys())[0]].shape[1]
            if len(set(data['future'].keys()).intersection(set(var_names))) == 0:
                var_names += list(data['future'].keys())
            for time_type, value_dicts in data.items():
                x_new[time_type] = dict()
                for var_name, value_df in value_dicts.items():
                    if var_name in var_names:
                        if time_type=='past':
                            var_lag_ids = [l for l in range(past_shape) if l in lag_ids]
                        else:
                            var_lag_ids = [l for l in range(future_shape) if (l + past_shape) in lag_ids]
                        cols = value_df.columns[var_lag_ids]
                        x_new[time_type][var_name] = data[time_type][var_name][cols]
            return x_new
        calendar_vars = {'month', 'dayweek', 'hour', 'minute'}
        if not isinstance(data, dict):
            feature_selector = self.feature_selectors[selector_tag]
            if transform_calendar:
                for var_name in calendar_vars:
                    if any([var_name == col or f'{var_name}_lag' in col for col in feature_selector['names']]):
                        var_cal = []
                        for c in [col for col in feature_selector['names'] if var_name == col or f'{var_name}_lag' in col ]:
                            var_cal += [c.replace(c.split('_')[0], f'{c.split("_")[0]}_sin'),
                                        c.replace(c.split('_')[0], f'{c.split("_")[0]}_cos')]
                        feature_selector['names'] = np.array([f for f in feature_selector['names']
                                                              if f not in [col for col in feature_selector['names']
                                                                           if var_name == col or f'{var_name}_lag' in col ]]
                                                             + var_cal)
            data_new = data[feature_selector['names']] if len(feature_selector['names']) > 0 else data
        else:
            data_new = dict()
            for group in data.keys():
                feature_selector = self.feature_selectors[f'{selector_tag }_{group}']
                if transform_calendar:
                    for var_name in calendar_vars:
                        if any([var_name == col or f'{var_name}_lag' in col  for col in feature_selector['names']]):
                            var_cal = []
                            for c in [col for col in feature_selector['names'] if var_name == col or f'{var_name}_lag' in col ]:
                                var_cal += [c.replace(c.split('_')[0], f'{c.split("_")[0]}_sin'),
                                           c.replace(c.split('_')[0], f'{c.split("_")[0]}_cos')]
                            feature_selector['names'] =  np.array([f for f in feature_selector['names']
                                                              if f not in [col for col in feature_selector['names']
                                                                           if var_name == col or f'{var_name}_lag' in col ]]
                                                             + var_cal)
                data_new[group] = data[group][feature_selector['names']] if len(feature_selector['names']) > 0 else data[group]
        return data_new
