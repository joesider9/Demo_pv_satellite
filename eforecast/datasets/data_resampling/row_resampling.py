import copy
import numpy as np
import pandas as pd

from numpy.random import SeedSequence, default_rng

import dask
import dask.dataframe as dd

from sklearn.neighbors import NearestNeighbors


class RowResampler:

    def __init__(
            self,
            sampling_strategy="auto",
            random_state=None,
            n_jobs=None,
    ):
        self.X_lstm_class = None
        self.nn_ = None
        self.radius = None
        self.X_class = None
        self.sampling_strategy = sampling_strategy
        self.random_state = random_state
        self.n_jobs = n_jobs

    def resample(self, row):
        random_state = row['random_state']
        threshold = self.radius.copy(deep=True).values
        x = row.drop(index=['random_state']).to_frame().T
        nns = self.nn_.kneighbors(x.values, return_distance=False)[0, 1:]
        dist = np.abs(self.X_class.iloc[nns].values - x.values)
        ind = np.where(np.all(dist <= threshold, axis=1))[0]
        dates_ind = self.X_class.index[ind]
        while dates_ind.shape[0] > 16:
            threshold -= 0.001 * threshold
            ind = np.where(np.all(dist <= threshold, axis=1))[0]
            dates_ind = self.X_class.index[ind]
            if (threshold < 0.001).all() and dates_ind.shape[0] > 16:
                dates_ind = dates_ind[:16]
        while dates_ind.shape[0] < 4:
            threshold += 0.001 * threshold
            ind = np.where(np.all(dist <= threshold, axis=1))[0]
            dates_ind = self.X_class.index[ind]
        nns = self.X_class.loc[dates_ind]
        quantile = random_state.uniform(low=0.05, high=0.95, size=1)
        sample = np.quantile(nns.values, q=quantile, axis=0)
        sample = pd.DataFrame(sample, index=[row.name], columns=self.X_class.columns)
        if self.X_lstm_class is not None:
            nns_lstm = self.X_lstm_class.loc[dates_ind]
            sample_lstm = np.quantile(nns_lstm.values, q=quantile, axis=0)
            sample_lstm = pd.DataFrame(sample_lstm, index=[row.name], columns=self.X_lstm_class.columns)
            sample = pd.concat([sample, sample_lstm], axis=1)
        return sample

    def fit_resample(self, X, y, X_lstm=None):
        ss = SeedSequence(12345)
        X_resampled = pd.DataFrame(index=X.index, columns=X.columns)
        if X_lstm is not None:
            X_lstm_flat = X_lstm.reshape([-1, X_lstm.shape[1] * X_lstm.shape[2]])
            cols_X_lstm = [f'lstm_{i}' for i in range(X_lstm_flat.shape[1])]
            X_lstm_df = pd.DataFrame(X_lstm_flat, index=X.index, columns=cols_X_lstm)
            X_lstm_resampled = pd.DataFrame(index=X.index, columns=cols_X_lstm)
        else:
            X_lstm_resampled = None
            X_lstm_df = None
            cols_X_lstm = None

        for class_sample in np.unique(y):
            print(f'Resampling Class {class_sample}')
            target_class_indices = X.index[np.flatnonzero(y == class_sample)]
            self.X_class = X.loc[target_class_indices]
            if X_lstm_df is not None:
                self.X_lstm_class = X_lstm_df.loc[target_class_indices]
            else:
                self.X_lstm_class = None
            random_states = [default_rng(s) for s in ss.spawn(self.X_class.shape[0])]
            self.radius = 0.01 * self.X_class.shape[1] * (self.X_class.max() - self.X_class.min())
            leaf_size = 20 if self.X_class.shape[0] > 200 else int(self.X_class.shape[0] / 10)
            self.nn_ = NearestNeighbors(n_neighbors=np.minimum(self.X_class.shape[0] - 1, 100), leaf_size=leaf_size)
            self.nn_.fit(self.X_class)
            X_class = self.X_class.copy(deep=True)
            X_class['random_state'] = random_states

            ddf = dd.from_pandas(X_class, npartitions=np.minimum(self.X_class.shape[0], self.n_jobs))
            ddf_update = ddf.map_partitions(lambda df: df.apply(self.resample, axis=1),
                                            meta=pd.DataFrame).compute(scheduler='processes')
            res = ddf_update.to_list()
            res = pd.concat(res)
            if X_lstm_resampled is not None:
                res_lstm = res[cols_X_lstm]
                X_lstm_resampled.loc[target_class_indices] = res_lstm
                res = res.drop(columns=cols_X_lstm)
            X_resampled.loc[target_class_indices] = res
            self.nn_ = None
            self.radius = None
            self.X_class = None
            X_class = None
        if X_lstm is None:
            return X_resampled
        else:
            X_lstm_resampled = X_lstm_resampled.values.reshape([-1, X_lstm.shape[1], X_lstm.shape[2]])
            return X_resampled, X_lstm_resampled

