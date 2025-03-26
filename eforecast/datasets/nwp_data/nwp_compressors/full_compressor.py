import numpy as np
import pandas as pd


class FullCompressor:
    def __init__(self, static_data, nwp_data, nwp_metadata):
        self.static_data = static_data
        self.horizon = self.static_data['horizon']
        self.use_data_before_and_after_target = self.static_data['use_data_before_and_after_target']
        self.type = self.static_data['type']
        self.nwp_metadata = nwp_metadata
        self.nwp_data = nwp_data

    def full_compressor(self, data, ax):
        full_type = self.full_type(ax)
        if full_type == 'coarse':
            names, data_compressed = self.full_coarse(data)
        elif full_type == 'reduced':
            names, data_compressed = self.full_reduced(data)
        elif full_type == 'detailed':
            names, data_compressed = self.full_detailed(data)
        else:
            raise ValueError('Unknown full_type')
        return names, data_compressed

    def perform_full_compress(self, i, ax, nwp_data):
        ax_name = '_'.join(ax)
        names, data = self.full_compressor(nwp_data[:, i, :, :], ax)
        data = pd.DataFrame(data, index=self.nwp_metadata['dates'],
                            columns=[ax_name + '_' + name for name in names])
        return data

    def full_type(self, ax):
        var_name, lag = ax[1].split('_')

        if self.horizon == 0:
            if self.use_data_before_and_after_target:
                lags = [-1, 0, 1]
            else:
                lags = [0]
        else:
            if self.use_data_before_and_after_target:
                lags = [-1] + [i for i in range(self.horizon + 1)]
            else:
                lags = [i for i in range(self.horizon)]
        if (var_name == 'WS' and self.type == 'wind') or (var_name == 'Flux' and self.type == 'pv'):
            if (lag == lags[0] or lag == lags[-1]) and self.use_data_before_and_after_target:
                return 'reduced'
            else:
                return 'detailed'
        elif var_name in {'WD', 'Cloud'}:
            return 'detailed'
        else:
            return 'coarse'

    def full_reduced(self, data):
        if self.static_data['regional']:
            shape = data.shape
            data = data.reshape(-1, np.prod(shape[1:]))
            data_full = np.hstack(
                (np.mean(data, axis=1).reshape(-1, 1), np.percentile(data, [5, 25, 50, 75, 95], axis=1).T))
            names = ['mean'] + ['pcl_' + str(pcl) for pcl in [5, 25, 50, 75, 95]]
            return names, data_full
        else:
            if data.shape[-1] == 5 and data.shape[-2] == 5:
                data_center = data[:, 2, 2]
                names = ['center']
                ind = np.array([[1, j] for j in range(1, 5)]
                               + [[i, 1] for i in range(2, 5)]
                               + [[2, 3], [3, 2], [3, 3], [2, 2]]
                               )
                data_close = data[:, ind[:, 0], ind[:, 1]].reshape(-1, 1)
                data_close = np.percentile(data_close, [5, 50, 95]).T
                names += ['c_pcl_' + str(pcl) for pcl in [5, 50, 95]]

                ind = np.array([[0, j] for j in range(5)]
                               + [[i, 0] for i in range(1, 5)]
                               + [[4, j] for j in range(1, 5)]
                               + [[i, 4] for i in range(1, 5)])
                data_around = data[:, ind[:, 0], ind[:, 1]].reshape(-1, 1)
                data_around = np.percentile(data_around, [5, 50, 95]).T
                data_full = np.hstack((data_center, data_close, data_around))
                names += ['pcl_' + str(pcl) for pcl in [5, 50, 95]]
                return names, data_full
            else:
                raise ValueError('The size of nwp data should be 5x5')

    def full_detailed(self, data):
        if self.static_data['regional']:
            shape = data.shape
            data_flat = data.reshape(-1, np.prod(shape[1:]))
            data_full = np.hstack(
                (np.mean(data, axis=-1), np.std(data, axis=-1),
                 np.percentile(data_flat, [5, 25, 50, 75, 95], axis=1).T))
            names = [f'mean_{i}' for i in range(shape[-2])] + [f'std_{i}' for i in range(shape[-2])] \
                    + ['pcl_' + str(pcl) for pcl in [5, 25, 50, 75, 95]]
            return names, data_full
        else:
            if data.shape[-1] == 5 and data.shape[-2] == 5:
                shape = data.shape
                data_full = data.reshape(-1, np.prod(shape[1:]))
                names = [f'pos_{i}_{j}' for j in range(shape[-1]) for i in range(shape[-2])]
                return names, data_full
            else:
                raise ValueError('The size of nwp data should be 5x5')

    def full_coarse(self, data):
        if self.static_data['regional']:
            shape = data.shape
            data = data.reshape(-1, np.prod(shape[1:]))
            full_data = np.hstack((np.mean(data, axis=1).reshape(-1, 1), np.percentile(data, [5, 50, 95], axis=1).T))
            names = ['mean'] + ['pcl_' + str(pcl) for pcl in [5, 50, 95]]
            return names, full_data
        else:
            if data.shape[-1] == 5 and data.shape[-2] == 5:
                centered = data[:, 2, 2]
                shape = data.shape
                data = data.reshape(-1, np.prod(shape[1:]))
                full_data = np.hstack((centered.reshape(-1, 1), np.percentile(data, [5, 50, 95], axis=1).T))
                names = ['center'] + ['pcl_' + str(pcl) for pcl in [5, 50, 95]]
                return names, full_data
            else:
                raise ValueError('The size of nwp data should be 5x5')

    def full_compress(self):
        groups = self.nwp_metadata['groups']
        axis = self.nwp_metadata['axis']
        if len(groups) == 0:
            nwp_compressed = pd.DataFrame()
            for var_id in range(len(axis)):
                for i, ax in enumerate(axis[var_id]):
                    data = self.perform_full_compress(i, ax, self.nwp_data[..., var_id])
                    nwp_compressed = pd.concat([nwp_compressed, data], axis=1)
            nwp_compressed_all = nwp_compressed
        else:
            nwp_compressed = dict()
            nwp_compressed_distributed = dict()
            for group in groups:
                group_name = '/'.join(group) if isinstance(group, tuple) else group
                nwp_compressed[group_name] = pd.DataFrame()
                for var_id in range(len(axis[group_name])):
                    for i, ax in enumerate(axis[group_name][var_id]):
                        data = self.perform_full_compress(i, ax, self.nwp_data[group_name][..., var_id])
                        nwp_compressed[group_name] = pd.concat([nwp_compressed[group_name], data], axis=1)
            nwp_compressed_all = pd.DataFrame()
            for group_name, data in nwp_compressed.items():
                nwp_compressed_all = pd.concat([nwp_compressed_all, data], axis=1)
        return nwp_compressed_all, nwp_compressed
