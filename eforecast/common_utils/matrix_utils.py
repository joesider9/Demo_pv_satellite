import cv2

import numpy as np


def rescale(arr, n_rows, n_col):
    return cv2.resize(arr, (n_col, n_rows), interpolation=cv2.INTER_AREA)


def stack_sample(x, sample):
    if x.shape[0] == 0:
        x = np.expand_dims(sample, axis=0)
    else:
        x = np.vstack((x, np.expand_dims(sample, axis=0)))
    return x


def stack_batch(x, sample):
    if x.shape[0] == 0:
        x = sample
    else:
        x = np.vstack((x, sample))
    return x
