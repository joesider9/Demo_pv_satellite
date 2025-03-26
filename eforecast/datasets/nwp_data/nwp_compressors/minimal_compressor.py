import numpy as np
import pandas as pd


class MinimalCompressor:
    def __init__(self, static_data, nwp_data, nwp_metadata):
        self.static_data = static_data
        self.horizon = self.static_data['horizon']
        self.use_data_before_and_after_target = self.static_data['use_data_before_and_after_target']
        self.type = self.static_data['type']
        self.nwp_metadata = nwp_metadata
        self.nwp_data = nwp_data

    def dense_compressor(self, data, ax):
        names, data_compressed = self.dense_coarse(data)
        return names, data_compressed

    def perform_dense_compress(self, i, ax, nwp_data):
        ax_name = '_'.join(ax)
        names, data = self.dense_compressor(nwp_data[:, i, :, :], ax)
        data = pd.DataFrame(data, index=self.nwp_metadata['dates'],
                            columns=[ax_name + '_' + name for name in names])
        return data


    def dense_coarse(self, data):
        if self.static_data['regional']:
            shape = data.shape
            data = data.reshape(-1, np.prod(shape[1:]))
            dense_data = np.hstack((np.mean(data, axis=1).reshape(-1, 1), np.percentile(data, [5, 50, 95], axis=1).T))
            names = ['mean'] + ['pcl_' + str(pcl) for pcl in [5, 50, 95]]
            return names, dense_data
        else:
            if data.shape[-1] == 5 and data.shape[-2] == 5:
                dense_data = data[:, 2:4, :][:, :, 2:4].reshape(-1, 4)
                names = ['center' + str(pcl) for pcl in range(4)]
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
