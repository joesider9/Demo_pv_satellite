import numpy as np
import pandas as pd


class LoadCompressor:
    def __init__(self, static_data, nwp_data, nwp_metadata):
        self.static_data = static_data
        self.horizon = self.static_data['horizon']
        self.use_data_before_and_after_target = self.static_data['use_data_before_and_after_target']
        self.type = self.static_data['type']
        self.nwp_metadata = nwp_metadata
        self.nwp_data = nwp_data


    def load_compressor(self, data):
        shape = data.shape
        data = data.reshape(-1, np.prod(shape[1:]))
        return np.mean(data, axis=1)

    def perform_load_compress(self, i, ax, nwp_data, group_name=None):
        ax_name = '_'.join(ax)
        variable = ax[1].split('_')[0]
        data = self.load_compressor(nwp_data[:, :, :, :, i])
        data = pd.DataFrame(data, index=self.nwp_metadata['dates'], columns=[ax_name])
        col = f'{variable}_max' if group_name is None else '_'.join([f'{variable}_max', group_name])
        data[col] = data.groupby(by=pd.DatetimeIndex(data.index.date))[ax_name].max()
        col = f'{variable}_min' if group_name is None else '_'.join([f'{variable}_min', group_name])
        data[col] = data.groupby(by=pd.DatetimeIndex(data.index.date))[ax_name].min()
        col = f'{variable}_mean' if group_name is None else '_'.join([f'{variable}_mean', group_name])
        data[col] = data.groupby(by=pd.DatetimeIndex(data.index.date))[ax_name].mean()
        col = f'{variable}_std' if group_name is None else '_'.join([f'{variable}_std', group_name])
        data[col] = data.groupby(by=pd.DatetimeIndex(data.index.date))[ax_name].mean()
        data = data.ffill()
        data = data.bfill()

        return data

    def load_compress(self):
        groups = self.nwp_metadata['groups']
        axis = self.nwp_metadata['axis']
        if len(groups) == 0:
            nwp_compressed = pd.DataFrame()
            for var_id in range(len(axis)):
                for i, ax in enumerate(axis[var_id]):
                    data = self.perform_load_compress(i, ax, self.nwp_data[..., var_id])
                    nwp_compressed = pd.concat([nwp_compressed, data], axis=1)
            nwp_compressed_all = nwp_compressed
        else:
            nwp_compressed = dict()
            for group in groups:
                group_name = '/'.join(group) if isinstance(group, tuple) else group
                nwp_compressed[group_name] = pd.DataFrame()
                for var_id in range(len(axis[group_name])):
                    for i, ax in enumerate(axis[group_name][var_id]):
                        data = self.perform_load_compress(i, ax, self.nwp_data[group_name][..., var_id],
                                                          group_name=group_name)
                        nwp_compressed[group_name] = pd.concat([nwp_compressed[group_name], data], axis=1)
            nwp_compressed_all = pd.DataFrame()
            for group_name, data in nwp_compressed.items():
                nwp_compressed_all = pd.concat([nwp_compressed_all, data], axis=1)
            extra_temp_vars = list(set([col.split('_')[0] for col in nwp_compressed_all.columns]))
            for extra_var in extra_temp_vars:
                cols = []
                for group_name in nwp_compressed.keys():
                    cols += [col for col in nwp_compressed_all.columns if extra_var in col and group_name in col]
                nwp_compressed_all[extra_var] = nwp_compressed_all[cols].mean(axis=1)

        return nwp_compressed_all, nwp_compressed
