import random

import numpy as np
import pandas as pd

from eforecast.datasets.files_manager import FilesManager
from eforecast.common_utils.dataset_utils import sync_datasets


class Splitter:
    def __init__(self, static_data, is_online=False, train=False):
        self.static_data = static_data
        self.is_online = is_online
        self.train = train
        self.is_fuzzy = self.static_data['is_Fuzzy']
        self.is_Global = self.static_data['is_Global']
        self.val_test_ratio = self.static_data['val_test_ratio']
        self.file_manager = FilesManager(static_data, is_online=is_online, train=train)

    def split_cluster_data(self, clustered_dates, cv_mask):
        group_data = dict()
        dates = cv_mask[0].union(cv_mask[1]).union(cv_mask[2])
        clustered_dates['global'] = dates
        N = dates.shape[0]
        days_all, n_frames = self.group_dates(dates)
        group_data['global'] = {'days': pd.DatetimeIndex(list(days_all.keys())),
                                'days_all': days_all,
                                'N': N,
                                'sel_dates': pd.DatetimeIndex([]),
                                'sel_dates_all': pd.DatetimeIndex([])}

        for rule_name, rule_dates in clustered_dates.items():
            days_rule, n_frames = self.group_dates(rule_dates)
            group_data[rule_name] = {'days': pd.DatetimeIndex(list(days_rule.keys())),
                                     'days_all': days_rule,
                                     'N': rule_dates.shape[0],
                                     'sel_dates': pd.DatetimeIndex([]),
                                     'sel_dates_all': pd.DatetimeIndex([])}
        min_ratio = 2 * self.val_test_ratio
        max_ratio = 2 * self.val_test_ratio
        weights_dict = {name: value['sel_dates_all'].shape[0] / value['N'] for name, value in group_data.items()}
        weights = np.array(list(weights_dict.values()))
        flags = [w < max_ratio and w > min_ratio for w in weights]
        trials = 0
        while not all(flags) and trials < 100:
            if trials % 4 == 0:
                min_ratio -= 0.01
                max_ratio += 0.01
            trials += 1
            sel_dates = pd.DatetimeIndex([])
            for name in group_data.keys():
                group_data[name]['sel_dates'] = pd.DatetimeIndex([])
                group_data[name]['sel_dates_all'] = pd.DatetimeIndex([])
            weights_dict = {name: value['sel_dates_all'].shape[0] / value['N'] for name, value in group_data.items()}
            weights = np.array(list(weights_dict.values()))
            flags_min = [w > min_ratio for w in weights]
            while not all(flags_min):
                i = np.random.randint(0, 100)
                rng = np.random.RandomState(i)
                group_data = dict(
                    sorted(group_data.items(), key=lambda item: item[1]['sel_dates_all'].shape[0], reverse=False))
                rule = list(group_data.keys())[0]
                diff_sets = group_data[rule]['days'].difference(sel_dates)
                if len(diff_sets) == 0:
                    break
                date = pd.to_datetime(rng.choice(diff_sets))
                sel_dates = sel_dates.append(pd.DatetimeIndex([date]))
                for name in group_data.keys():
                    if date in group_data[name]['days']:
                        group_data[name]['sel_dates'] = group_data[name]['sel_dates'].append(pd.DatetimeIndex([date]))
                        group_data[name]['sel_dates_all'] = (group_data[name]['sel_dates_all'].
                                                             append(group_data[name]['days_all'][date]))
                weights_dict = {name: value['sel_dates_all'].shape[0] / value['N'] for name, value in
                                group_data.items()}
                weights = np.array(list(weights_dict.values()))
                flags_min = [w > 2 * self.val_test_ratio for w in weights]
            flags = [w < max_ratio and w > min_ratio for w in weights]
        cluster_cvs = dict()
        for name, value in group_data.items():
            if value['sel_dates_all'].shape[0] == 0:
                raise ValueError('Better to run again clustering')
            half = int(value['sel_dates_all'].shape[0] / 2)
            dates_train = clustered_dates[name].difference(value['sel_dates_all']).sort_values()
            dates_val = value['sel_dates_all'][:half].sort_values()
            dates_test = value['sel_dates_all'][half:].sort_values()
            cluster_cvs[name] = [dates_train, dates_val, dates_test]
        return cluster_cvs

    def split(self, refit=False):
        if refit:
            self.file_manager.remove_cv_data_files()
        cv_mask = self.file_manager.check_if_exists_cv_data()
        if cv_mask is None:
            data_row = self.file_manager.check_if_exists_row_data()
            data_row = data_row['calendar'].dropna(axis='index')
            if data_row is None:
                raise ImportError('Cannot find data row to split. Check if data are exists Or choose another dataset')
            y = self.file_manager.check_if_exists_target().dropna(axis='index')
            if y is None:
                raise ImportError('Cannot find target data to stratify for split. Check if data are exists')
            data_row, y = sync_datasets(data_row, y)
            dates = y.index
            if self.static_data['ts_resolution'] != 'D':
                days = [g.index for n, g in y.groupby(pd.Grouper(freq='D'))]
                period = data_row.shape[0] / len(days)
            else:
                days = [g.index for n, g in y.groupby(pd.Grouper(freq='W'))]
                period = data_row.shape[0] / len(days)
            n_frames = 2 * int((data_row.shape[0] * self.val_test_ratio) / period)
            indices = random.sample(list(range(len(days))), n_frames)
            mask_train = pd.DatetimeIndex([])
            mask_val = pd.DatetimeIndex([])
            mask_test = pd.DatetimeIndex([])
            count = 0
            for i in range(len(days)):
                if i in indices:
                    count += 1
                    if count < (n_frames / 2):
                        mask_val = mask_val.append(days[i])
                    else:
                        mask_test = mask_test.append(days[i])
                else:
                    mask_train = mask_train.append(days[i])
            mask_train = mask_train.sort_values()
            mask_val = mask_val.sort_values()
            mask_test = mask_test.sort_values()
            self.file_manager.save_cv_data([mask_train, mask_val, mask_test])

    def group_dates(self, dates):
        if self.static_data['ts_resolution'] != 'D':
            days = {n: g.index for n, g in
                     pd.DataFrame(list(range(dates.shape[0])), index=dates).groupby(pd.Grouper(freq='D'))}
        else:
            days = {n: g.index for n, g in
                    pd.DataFrame(list(range(dates.shape[0])), index=dates).groupby(pd.Grouper(freq='W'))}
        period = dates.shape[0] / len(days)
        n_frames = 2 * int((dates.shape[0] * self.val_test_ratio) / period)
        return days, n_frames