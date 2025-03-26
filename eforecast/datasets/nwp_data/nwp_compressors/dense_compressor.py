import numpy as np
import pandas as pd


class DenseCompressor:
    def __init__(self, static_data, nwp_data, nwp_metadata):
        self.static_data = static_data
        self.horizon = self.static_data['horizon']
        self.use_data_before_and_after_target = self.static_data['use_data_before_and_after_target']
        self.type = self.static_data['type']
        self.nwp_metadata = nwp_metadata
        self.nwp_data = nwp_data

    def dense_compressor(self, data, ax):
        dense_type = self.dense_type(ax)
        if dense_type == 'coarse':
            names, data_compressed = self.dense_coarse(data)
        elif dense_type == 'reduced':
            names, data_compressed = self.dense_reduced(data)
        elif dense_type == 'detailed':
            names, data_compressed = self.dense_detailed(data)
        else:
            raise ValueError('Unknown dense_type')
        return names, data_compressed

    def perform_dense_compress(self, i, ax, nwp_data):
        ax_name = '_'.join(ax)
        names, data = self.dense_compressor(nwp_data[:, i, :, :], ax)
        data = pd.DataFrame(data, index=self.nwp_metadata['dates'],
                            columns=[ax_name + '_' + name for name in names])
        return data

    def dense_type(self, ax):
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
            if int(lag) == 0 and self.horizon == 0:
                return 'detailed'
            elif int(lag) == 1 and self.use_data_before_and_after_target and self.horizon == 0:
                return 'coarse'
            elif int(lag) > 0 and self.horizon > 0:
                return 'reduced'
            else:
                return 'coarse'
        elif var_name in {'WD', 'Cloud'}:
            return 'reduced'
        else:
            return 'coarse'

    def dense_reduced(self, data):
        if self.static_data['regional']:
            shape = data.shape
            data = data.reshape(-1, np.prod(shape[1:]))
            data_dense = np.hstack(
                (np.mean(data, axis=1).reshape(-1, 1), np.std(data, axis=1).reshape(-1, 1),
                 np.percentile(data, [5, 95], axis=1).T))
            names = ['mean'] + ['std'] + ['pcl_' + str(pcl) for pcl in [5, 95]]
            return names, data_dense
        else:
            if data.shape[-1] == 5 and data.shape[-2] == 5:
                data_center = data[:, 2, 2].reshape(-1, 1)
                names = ['center']
                ind = np.array([[1, 1], [1, 2], [1, 3],
                                [2, 1], [2, 2], [2, 3],
                                [3, 1], [3, 2], [3, 3]]
                               )
                data_close = data[:, ind[:, 0], ind[:, 1]]
                data_close = np.percentile(data_close, [5, 50, 95], axis=1).T
                names += ['c_pcl_' + str(pcl) for pcl in [5, 50, 95]]

                ind = np.array([[0, j] for j in range(5)]
                               + [[i, 0] for i in range(1, 5)]
                               + [[4, j] for j in range(1, 5)]
                               + [[i, 4] for i in range(1, 5)])
                data_around = data[:, ind[:, 0], ind[:, 1]]
                data_around = np.hstack((np.mean(data_around, axis=1).reshape(-1, 1),
                                         np.std(data_around, axis=1).reshape(-1, 1)))
                data_dense = np.hstack((data_center, data_close, data_around))
                names += ['mean', 'std']
                return names, data_dense
            else:
                raise ValueError('The size of nwp data should be 5x5')

    def dense_detailed(self, data):
        if self.static_data['regional']:
            shape = data.shape
            data = data.reshape(-1, np.prod(shape[1:]))
            data_dense = np.hstack(
                (np.mean(data, axis=1).reshape(-1, 1), np.percentile(data, [5, 25, 75, 95], axis=1).T))
            names = ['mean'] + ['pcl_' + str(pcl) for pcl in [5, 25, 75, 95]]
            return names, data_dense
        else:
            if data.shape[-1] == 5 and data.shape[-2] == 5:
                data_center = data[:, 2, 2].reshape(-1, 1)
                names = ['center']
                ind = np.array([[1, 1], [1, 2], [1, 3],
                                [2, 1], [2, 3],
                                [3, 1], [3, 2], [3, 3]]
                               )
                data_center1 = data[:, ind[:, 0], ind[:, 1]]
                data_center1 = np.percentile(data_center1, [5, 25, 50, 75, 95], axis=1).T
                names += ['centered_' + str(pcl) for pcl in [5, 25, 50, 75, 95]]
                ind = np.array([[0, j] for j in range(5)]
                               + [[i, 0] for i in range(1, 5)]
                               + [[4, j] for j in range(1, 5)]
                               + [[i, 4] for i in range(1, 5)])
                data_around = data[:, ind[:, 0], ind[:, 1]]
                data_around = np.hstack((np.mean(data_around, axis=1).reshape(-1, 1),
                                         np.std(data_around, axis=1).reshape(-1, 1)))
                data_dense = np.hstack((data_center, data_center1, data_around))
                names += ['mean', 'std']
                return names, data_dense
            else:
                raise ValueError('The size of nwp data should be 5x5')

    def dense_coarse(self, data):
        if self.static_data['regional']:
            shape = data.shape
            data = data.reshape(-1, np.prod(shape[1:]))
            dense_data = np.hstack((np.mean(data, axis=1).reshape(-1, 1), np.percentile(data, [5, 50, 95], axis=1).T))
            names = ['mean'] + ['pcl_' + str(pcl) for pcl in [5, 50, 95]]
            return names, dense_data
        else:
            if data.shape[-1] == 5 and data.shape[-2] == 5:
                centered = data[:, 2, 2]
                shape = data.shape
                data = data.reshape(-1, np.prod(shape[1:]))
                data_around = np.hstack((np.mean(data, axis=1).reshape(-1, 1),
                                         np.std(data, axis=1).reshape(-1, 1)))
                dense_data = np.hstack((centered.reshape(-1, 1), data_around))
                names = ['center'] + ['mean', 'std']
                return names, dense_data
            else:
                raise ValueError('The size of nwp data should be 5x5')

    def dense_compress(self):
        groups = self.nwp_metadata['groups']
        axis = self.nwp_metadata['axis']
        if len(groups) == 0:
            nwp_compressed = pd.DataFrame()
            for var_id in range(len(axis)):
                for i, ax in enumerate(axis[var_id]):
                    data = self.perform_dense_compress(i, ax, self.nwp_data[..., var_id])
                    nwp_compressed = pd.concat([nwp_compressed, data], axis=1)
            nwp_compressed_all = nwp_compressed
        else:
            nwp_compressed = dict()
            for group in groups:
                group_name = '/'.join(group) if isinstance(group, tuple) else group
                nwp_compressed[group_name] = pd.DataFrame()
                for var_id in range(len(axis[group_name])):
                    for i, ax in enumerate(axis[group_name][var_id]):
                        data = self.perform_dense_compress(i, ax, self.nwp_data[group_name][..., var_id])
                        nwp_compressed[group_name] = pd.concat([nwp_compressed[group_name], data], axis=1)
            nwp_compressed_all = pd.DataFrame()
            for group_name, data in nwp_compressed.items():
                nwp_compressed_all = pd.concat([nwp_compressed_all, data], axis=1)
        return nwp_compressed_all, nwp_compressed
