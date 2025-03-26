import numpy as np
import pandas as pd


def transform_rated(rated, y):
    if len(y.shape) > 1:
        if y.shape[1] > 1:
            rated = y.values if rated is None else rated
        else:
            rated = y.values.ravel() if rated is None else rated
    else:
        rated = y.values if rated is None else rated
    return rated


def flat_target(targ):
    if len(targ.shape) == 2:
        if targ.shape[1] == 1:
            targ = targ.ravel()
    return targ


def compute_metrics(pred, y, rated, predictor_name, multi_output=False):
    rated = transform_rated(rated, y)
    y_np = flat_target(y.values)
    pred_np = flat_target(pred.values)
    err = np.abs(pred_np - y_np) / rated
    if not multi_output:
        sse = np.sum(np.square(pred_np - y_np))
        rms = np.sqrt(np.nanmean(np.square(err)))
        mae = np.nanmean(err)
        mse = sse / y_np.shape[0]
        return pd.DataFrame(np.array([sse, 100 * rms, 100 * mae, mse]).reshape(1, -1),
                            columns=['sse', 'rms', 'mae', 'mse'],
                            index=[predictor_name])
    else:
        mae = np.nanmean(err, axis=0).reshape(1, -1)
        res = pd.DataFrame(100 * mae, columns=[f'hour_ahead_{i}' for i in range(err.shape[1])],
                           index=[predictor_name])
        res['average'] = res.mean(axis=1)
        return res
