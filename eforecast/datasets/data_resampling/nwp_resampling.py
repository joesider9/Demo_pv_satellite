import copy
import numpy as np

from numpy.random import SeedSequence, default_rng

import dask
from dask.distributed import Client
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import _safe_indexing


class NWPResampler:

    def __init__(
            self,
            sampling_strategy="auto",
            random_state=None,
            n_jobs=None,
    ):
        self.nn_ = None
        self.client = Client(n_workers=n_jobs)
        self.sampling_strategy = sampling_strategy
        self.random_state = random_state
        self.n_jobs = n_jobs

    @staticmethod
    def resample(i, dist, nns, threshold_dist, random_state):
        threshold = copy.deepcopy(threshold_dist)
        ind = np.where(dist <= threshold)[0]
        while ind.shape[0] > 30:
            threshold -= 0.001 * threshold
            ind = np.where(dist <= threshold)[0]
            if threshold < 0.01:
                ind = ind[:30]
        while ind.shape[0] < 6:
            threshold += 0.001 * threshold
            ind = np.where(dist <= threshold)[0]
        n_neighbors = ind.shape[0]
        nns = nns[ind]
        np.random.shuffle(nns)
        cols = random_state.choice(n_neighbors, size=1)

        return nns[cols]

    def fit_resample(self, X_3d, y):
        ss = SeedSequence(12345)
        shape = X_3d.shape
        X = X_3d.reshape(-1, np.prod(shape[1:]))
        X_3d_resampled = np.zeros_like(X_3d)

        for class_sample in np.unique(y):
            print(f'Resampling Class {class_sample}')
            target_class_indices = np.flatnonzero(y == class_sample)
            n_samples = target_class_indices.shape[0]
            X_class = _safe_indexing(X, target_class_indices)
            X_class_3d = _safe_indexing(X_3d, target_class_indices)
            random_states = [default_rng(s) for s in ss.spawn(X_class.shape[0])]
            radius = 0.01 * X_class.shape[1] * (X_class.max() - X_class.min())
            leaf_size = 20 if X_class.shape[0] > 200 else int(X_class.shape[0] / 10)
            self.nn_ = NearestNeighbors(n_neighbors=np.minimum(X_class.shape[0] - 1, 100), leaf_size=leaf_size)
            self.nn_.set_params(**{"n_jobs": self.n_jobs})
            self.nn_.fit(X_class)
            dist, nns = self.nn_.kneighbors(X_class)
            dist = dist[:, 1:]
            nns = nns[:, 1:]
            futures = []
            for i in range(X_class.shape[0]):
                future = self.client.submit(self.resample, i, dist[i], nns[i], radius, random_states[i])
                futures.append(future)
            cols = self.client.gather(futures)
            diffs_3d = X_class_3d[np.concatenate(cols)] - X_class_3d
            steps = np.expand_dims(random_states[0].uniform(low=0.1, high=0.9, size=(diffs_3d.shape[0])),
                                   axis=[ndim for ndim in range(1, diffs_3d.ndim)])
            steps = np.tile(steps, diffs_3d.shape[1:]) + random_states[0].uniform(low=-0.1, high=0.1,
                                                                                  size=tuple(diffs_3d.shape))

            X_new_3d = X_class_3d + steps * diffs_3d
            X_3d_resampled[target_class_indices] = X_new_3d

        self.client.close()
        return X_3d_resampled
