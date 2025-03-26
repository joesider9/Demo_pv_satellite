import numpy as np
import pandas as pd

from numpy.random import SeedSequence, default_rng

import dask.dataframe as dd

from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KernelDensity
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression


class TargetResampler:

    def __init__(
            self,
            sampling_strategy="auto",
            random_state=None,
            n_jobs=None,
    ):
        self.data_resampled_class = None
        self.data_class = None
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
        y_swap = self.X_class.iloc[nns[0]].values
        dist = np.abs(self.data_class.iloc[nns].values - x.values)
        dates_ind = self.data_class.index[np.where(np.all(dist <= threshold, axis=1))[0]]
        while dates_ind.shape[0] > 30:
            threshold -= 0.001 * threshold
            dates_ind = self.data_class.index[np.where(np.all(dist <= threshold, axis=1))[0]]
            if (threshold < 0.01).all():
                dates_ind = dates_ind[:30]
        while dates_ind.shape[0] < 6:
            threshold += 0.001 * threshold
            dates_ind = self.data_class.index[np.where(np.all(dist <= threshold, axis=1))[0]]
        nns = self.X_class.loc[dates_ind]
        ks = KernelDensity(bandwidth=0.01, leaf_size=2)
        ks.fit(nns.values)
        y_ks = ks.sample(random_state=random_state.integers(1000))[0]

        pca = PCA(n_components=np.minimum(self.data_resampled_class.shape[1] - 1, np.minimum(dates_ind.shape[0] - 1, 10)))
        pca.fit(self.data_resampled_class.loc[dates_ind].values)
        X_tr = pca.transform(self.data_resampled_class.loc[dates_ind].values)
        if self.X_class.shape[1] > 1:
            lr = LinearRegression().fit(X_tr, self.X_class.loc[dates_ind].values)
            y_lr = lr.predict(pca.transform(x.values.reshape(1, -1))).ravel()
            return pd.DataFrame(np.array([y_swap, y_ks, y_lr]).reshape(1, -1), index=[row.name],
                                columns=[f'swap_{col}' for col in self.X_class.columns] +
                                        [f'kernel_density_{col}' for col in self.X_class.columns] +
                                        [f'linear_reg_{col}' for col in self.X_class.columns])
        else:
            lr = LinearRegression().fit(X_tr, self.X_class.loc[dates_ind].values.ravel())
            y_lr = lr.predict(pca.transform(x.values.reshape(1, -1)))
            if y_lr < np.min(self.X_class.loc[dates_ind].values.ravel()):
                y_lr = np.array([np.min(self.X_class.loc[dates_ind].values.ravel())])
            if y_lr > np.max(self.X_class.loc[dates_ind].values.ravel()):
                y_lr = np.array([np.max(self.X_class.loc[dates_ind].values.ravel())])
            return pd.DataFrame(np.array([y_swap, y_ks, y_lr]).reshape(1, -1), index=[row.name],
                                columns=['swap', 'kernel_density', 'linear_reg'])

    def fit_resample(self, X, data, data_resampled, y):
        ss = SeedSequence(12345)
        if X.shape[1] > 1:
            X_resampled = pd.DataFrame(index=X.index, columns=[f'swap_{col}' for col in X.columns] +
                                        [f'kernel_density_{col}' for col in X.columns] +
                                        [f'linear_reg_{col}' for col in X.columns])
        else:
            X_resampled = pd.DataFrame(index=X.index, columns=['swap', 'kernel_density', 'linear_reg'])
        for class_sample in np.unique(y):
            print(f'Resampling Class {class_sample}')
            target_class_indices = X.index[np.flatnonzero(y == class_sample)]
            self.X_class = X.loc[target_class_indices]
            self.data_class = data.loc[target_class_indices]
            self.data_resampled_class = data_resampled.loc[target_class_indices]
            random_states = [default_rng(s) for s in ss.spawn(self.X_class.shape[0])]
            self.radius = 0.01 * self.data_class.shape[1] * (self.data_class.max() - self.data_class.min())
            leaf_size = 20 if self.data_class.shape[0] > 200 else int(self.data_class.shape[0] / 10)
            self.nn_ = NearestNeighbors(n_neighbors=np.minimum(self.data_class.shape[0] - 1, 100), leaf_size=leaf_size)
            self.nn_.fit(self.data_class)
            data_resampled_class = self.data_resampled_class.copy(deep=True)
            data_resampled_class['random_state'] = random_states

            ddf = dd.from_pandas(data_resampled_class, npartitions=np.minimum(self.X_class.shape[0], self.n_jobs))
            ddf_update = ddf.map_partitions(lambda df: df.apply(self.resample, axis=1),
                                            meta=pd.DataFrame).compute(scheduler='processes')
            res = ddf_update.to_list()
            X_resampled.loc[target_class_indices] = pd.concat(res)
            self.nn_ = None
            self.radius = None
            self.X_class = None
            self.data_class = None
            self.data_resampled_class = None
            X_class = None
        return X_resampled
