import pandas as pd
import tqdm

import numpy as np

from sklearn.neighbors import KNeighborsClassifier


def bcp_fit(X, y, weight_size):
    sigma = np.nanstd(y - X, axis=0).reshape(-1, 1)
    err = []
    preds = []
    w = np.ones([1, weight_size]) / weight_size
    count = 0
    for inp, targ in tqdm.tqdm(zip(X, y)):
        inp = inp.reshape(-1, 1)
        mask = ~np.isnan(inp)
        pred = np.matmul(w[mask.T], inp[mask])
        preds.append(pred)
        e = targ - pred
        err.append(e)

        p = np.exp(-1 * np.square((targ - inp[mask].T) / (np.sqrt(2 * np.pi) * sigma[mask])))
        p = p / sum(p)
        w[mask.T] = ((w[mask.T] * p) / np.sum(w[mask.T] * p))
        w[np.where(w < 0)] = 0
        w /= np.sum(w)

        count += 1
    return w


def kmeans_fit(kmeans_model, labels, X, y):
    probs = dict()
    for label in np.unique(labels):
        probs[str(label)] = dict()
        ind = np.where(labels == label)[0]
        if ind.shape[0] > 10:
            x_ = X[ind]
            y_ = y[ind]
            best_x = np.argmin(np.abs(x_ - y_), axis=1)
            predictors = [predictor for predictor in range(x_.shape[1]) if predictor in best_x]
            p = KNeighborsClassifier()
            try:
                p.fit(x_, best_x)
            except:
                p = [1 / x_.shape[1] for predictor in range(x_.shape[1])]
                predictors = [predictor for predictor in range(x_.shape[1])]
        else:
            p = [1 / X.shape[1] for predictor in range(X.shape[1])]
            predictors = [predictor for predictor in range(X.shape[1])]
        probs[str(label)]['model'] = p
        probs[str(label)]['predictors'] = predictors

    model = {
        'probs': probs,
        'Kmean': kmeans_model
    }
    return model


def kmeans_predict(model, pred_methods, X_inputs, n_jobs):
    dates = pred_methods.index.intersection(X_inputs.index)
    pred = pred_methods.loc[dates]

    kmeans_model = model['Kmean']
    labels = kmeans_model.predict(X_inputs.loc[dates].values)

    probs = np.zeros([dates.shape[0], pred.shape[1]])
    for label in np.unique(labels):
        ind = np.where(labels == label)[0]
        knn = model['probs'][str(label)]['model']
        if isinstance(knn, list):
            pr = np.tile(np.array(knn).reshape(1, -1), [ind.shape[0], 1])
        else:
            pr = knn.predict_proba(pred.iloc[ind].values)
        probs[ind.reshape(-1, 1), np.array(model['probs'][str(label)]['predictors']).reshape(1, -1)] = pr

    probs_nan = np.copy(probs)
    probs_nan[probs_nan == 0] = np.nan
    probs_mn = np.nanmin(probs_nan, axis=1).reshape(-1, 1)
    probs_fl = (probs >= probs_mn).astype('int')
    probs = probs * probs_fl
    return np.sum(pred.values * probs / np.tile(np.sum(probs, axis=1).reshape(-1, 1), [1, pred.shape[1]]), axis=1)


def shallow_classifier_weighted_sum(proba, pred_methods, n_jobs):
    probs_nan = np.copy(proba)
    probs_nan[probs_nan == 0] = np.nan
    probs_mn = np.nanmin(probs_nan, axis=1).reshape(-1, 1)
    probs_fl = (proba >= probs_mn).astype('int')
    proba = proba * probs_fl
    return np.sum(pred_methods.values * proba / np.tile(np.sum(proba, axis=1).reshape(-1, 1),
                                                        [1, pred_methods.shape[1]]), axis=1)
