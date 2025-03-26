import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression

class RBFols:
    def __init__(self, rated, thres_act=0.001, n_clusters=None, width=None):
        self.err = None
        self.ncenters = n_clusters
        self.metrics = None
        self.model = {'centroids': None,
                      'Radius': None,
                      'W': None}
        self.rated = rated
        self.width = width
        self.n_clusters = n_clusters
        self.params = {'width': width,
                       'thres_act': thres_act,
                       'n_clusters': n_clusters}

    def _predict(self, model, x):
        v = (np.atleast_2d(x)[:, np.newaxis] - model['centroids'][np.newaxis, :]) / model['Radius']
        v = np.sqrt((v ** 2.).sum(-1))
        v = np.exp(-(v))
        v = (v - v.min()) / (v.max() - v.min())
        return v

    def compute_output(self, x, y, act_train, x_val, y_val, act_val):
        outs = y.shape[1]
        thres = np.concatenate([act_train, act_val]).max(axis=1).min() - 0.001
        lin_models = dict()
        for i in range(act_train.shape[1]):
            indices = np.where(act_train[:, i] >= thres)[0]
            if len(indices) != 0:
                X1 = x[indices]
                y1 = y[indices]
                lin_models[f'model_{i}'] = LinearRegression().fit(X1, y1)

        preds = np.nan * np.ones([act_val.shape[0], act_val.shape[1], y.shape[1]])
        for i in range(act_val.shape[1]):
            indices = np.where(act_val[:, i] >= thres)[0]
            if len(indices) != 0 and f'model_{i}' in lin_models.keys():
                X1 = x_val[indices]
                preds[indices, i, :] = lin_models[f'model_{i}'].predict(X1)
        pred = np.nanmean(preds, axis=1)
        pred[np.isnan(pred)] = 2
        pred[pred < 0] = 0
        return pred

    def compute_metrics(self, pred, y):
        ind = np.where(pred < 1)[0]
        ind2 = np.where(pred > 1)[0]
        if self.rated is None:
            self.rated = y[ind]
        else:
            self.rated = 1

        if y.shape[1] > 1:
            err = np.abs((pred[ind] - y[ind])) / self.rated
        else:
            err = np.abs(pred[ind].ravel() - y[ind].ravel()) / self.rated
        mae = np.mean(err)
        sse = np.sum(np.square(err))

        return mae, sse

    def _distance(self, obj_new, obj_old, obj_max, obj_min):
        if np.any(np.isinf(obj_old)):
            obj_old = obj_new.copy()
            obj_max = obj_new.copy()
            return True, obj_old, obj_max, obj_min
        if np.any(np.isinf(obj_min)) and not np.all(obj_max == obj_new):
            obj_min = obj_new.copy()
        d = 0
        for i in range(obj_new.shape[0]):
            if obj_max[i] < obj_new[i]:
                obj_max[i] = obj_new[i]
            if obj_min[i] > obj_new[i]:
                obj_min[i] = obj_new[i]
            if obj_max[i] - obj_min[i] == 0:
                d += (obj_new[i] - obj_old[i])
            else:
                d += (obj_new[i] - obj_old[i]) / (obj_max[i] - obj_min[i])
        if d < 0.0025:
            obj_old = obj_new.copy()
            return True, obj_old, obj_max, obj_min
        else:
            return False, obj_old, obj_max, obj_min

    def fit(self, I, O, X_val, y_val):
        obj_old = np.inf * np.ones(2)
        obj_max = np.inf * np.ones(2)
        obj_min = np.inf * np.ones(2)
        model = dict()
        model_temp = dict()

        k = np.sqrt(-np.log(0.5)) / self.width
        m, d = O.shape
        d *= m
        idx = np.arange(m)
        X = I.copy()
        P = np.exp(-(np.sqrt((((I[np.newaxis, :] - I[:, np.newaxis]) * k) ** 2.0).sum(-1))))
        G = np.array(P)
        D = (O * O).sum(0)
        e = ((np.matmul(P.T, O) ** 2.) / np.matmul((P * P).sum(0)[:, np.newaxis], D[np.newaxis, :])).sum(1)
        next = e.argmax()
        used = np.array([next])
        idx = np.delete(idx, next)
        W = P[:, next, np.newaxis]
        P = np.delete(P, next, 1)
        X = np.delete(X, next, 0)
        G1 = G[:, used]


        model_temp['centroids'] = I[used]
        model_temp['Radius'] = k
        act_train = self._predict(model_temp, I)
        act_val = self._predict(model_temp, X_val)
        pred_val = self.compute_output(I, O, act_train, X_val, y_val, act_val)
        mae_val, sse = self.compute_metrics(pred_val, y_val)
        metrics = np.array([mae_val, sse])

        flag, obj_old, obj_max, obj_min = self._distance(metrics, obj_old, obj_max, obj_min)

        remains = 0
        centers = I[used]
        ibias = k
        while len(used) < self.n_clusters and P.shape[1] > 0:
            wj = W[:, -1:]
            a = np.matmul(wj.T, P) / np.matmul(wj.T, wj)
            P = P - wj * a
            dist_all = ((X[np.newaxis, :] - I[used][:, np.newaxis]) ** 2.0).sum(-1).sum(0)
            e = ((np.matmul(P.T, O) ** 2.) / np.matmul((P * P).sum(0)[:, np.newaxis], D[np.newaxis, :])).sum(1)
            e[dist_all < dist_all.mean()] = 0
            next = e.argmax()
            W = np.append(W, P[:, next, np.newaxis], axis=1)

            dist1 = ((I[idx[next]][np.newaxis, :] - I[used][:, np.newaxis]) ** 2.0)
            dist = dist1.max(-1).squeeze()
            flag_dist = (dist < ((I.shape[1] / self.n_clusters) ** 2.0)).any()

            used = np.append(used, idx[next])

            P = np.delete(P, next, 1)
            X = np.delete(X, next, 0)
            idx = np.delete(idx, next)

            model_temp['centroids'] = I[used]
            model_temp['Radius'] = k
            act_train = self._predict(model_temp, I)
            act_val = self._predict(model_temp, X_val)
            pred_val = self.compute_output(I, O, act_train, X_val, y_val, act_val)
            mae_val, sse = self.compute_metrics(pred_val, y_val)
            metrics = np.array([mae_val, sse])
            print(f'MAE IS {mae_val} and obj is {obj_old[0]}' )
            flag, obj_old, obj_max, obj_min = self._distance(metrics, obj_old, obj_max, obj_min)
            if (not flag or flag_dist) and remains < 5 * self.n_clusters:
                used = used[:-1]
                remains += 1
            else:
                remains = 0
                centers = I[used]
                ibias = k
                obj_old = metrics.copy()

        model['centroids'] = centers
        model['Radius'] = ibias
        act_train = self._predict(model, I)
        act_val = self._predict(model, X_val)
        pred_val = self.compute_output(I, O, act_train, X_val, y_val, act_val)
        mae_val, sse = self.compute_metrics(pred_val, y_val)
        self.metrics = np.array([mae_val, sse])
        self.model = model
        self.err = mae_val
