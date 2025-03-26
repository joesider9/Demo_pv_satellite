import numpy as np
import pandas as pd


class RBFols:
    def __init__(self, rated, width=None, keep=None):
        self.err = None
        self.ncenters = None
        self.metrics = None
        self.model = {'centroids': None,
                      'Radius': None,
                      'W': None}
        self.rated = rated
        self.width = width
        self.keep = keep
        self.params = {'width': width,
                       'keep': keep}

    def get_params(self, deep=True):
        out = dict()
        for key, value in self.params.items():
            out[key] = value
        return out

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
            self.params[key] = value
        return self

    def predict(self, x):
        if isinstance(x, dict):
            for key in sorted(x.keys()):
                x[key].columns = [f'{key}_{col}' for col in x[key].columns]
            X_ = pd.concat([x[key] for key in sorted(x.keys())])
            x = X_.copy()
        return self._predict(self.model, x)

    def _predict(self, model, x):
        v = (np.atleast_2d(x)[:, np.newaxis] - model['centroids'][np.newaxis, :]) * model['Radius']
        v = np.sqrt((v ** 2.).sum(-1))
        v = np.exp(-(v ** 3.))
        if isinstance(model['W'], np.ndarray):
            v = np.matmul(v, model['W'][:-1]) + model['W'][-1]
        else:
            v = model['W'].predict(v)
        return v

    def compute_metrics(self, pred, y):
        if self.rated is None:
            self.rated = y.ravel()
        else:
            self.rated = 1
        if y.shape[1] > 1:
            err = np.abs((pred - y).ravel()) / self.rated
        else:
            err = np.abs(pred.ravel() - y.ravel()) / self.rated

        sse = np.sum(np.square(err))
        rms = np.sqrt(np.mean(np.square(err)))
        mae = np.mean(err)
        mse = sse / y.shape[0]

        return sse, rms, mae, mse

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
        if d < 0:
            obj_old = obj_new.copy()
            return True, obj_old, obj_max, obj_min
        else:
            return False, obj_old, obj_max, obj_min

    def fit(self, I, O, X_val, y_val):
        if isinstance(I, dict):
            for key in sorted(I.keys()):
                I[key].columns = [f'{key}_{col}' for col in I[key].columns]
            X_ = pd.concat([I[key] for key in sorted(I.keys())])
            I = X_.copy()
        obj_old = np.inf * np.ones(2)
        obj_max = np.inf * np.ones(2)
        obj_min = np.inf * np.ones(2)
        model = dict()
        model_temp = dict()
        remains = 0
        mse_best = np.inf
        k = np.sqrt(-np.log(0.5)) / self.width
        m, d = O.shape
        d *= m
        idx = np.arange(m)
        P = np.exp(-(np.sqrt((((I[np.newaxis, :] - I[:, np.newaxis]) * k) ** 2.0).sum(-1))) ** 3.0)
        G = np.array(P)
        D = (O * O).sum(0)
        e = ((np.matmul(P.T, O) ** 2.) / np.matmul((P * P).sum(0)[:, np.newaxis], D[np.newaxis, :])).sum(1)
        next = e.argmax()
        used = np.array([next])
        idx = np.delete(idx, next)
        W = P[:, next, np.newaxis]
        P = np.delete(P, next, 1)
        G1 = G[:, used]
        v = (np.atleast_2d(I)[:, np.newaxis] - I[used][np.newaxis, :]) * k
        v = np.sqrt((v ** 2.).sum(-1))
        v = np.exp(-(v ** 3.))
        try:
            out_layer = np.linalg.lstsq(np.hstack([v, np.ones([v.shape[0], 1])]), O, rcond=None)[0]
            # out_layer = DecisionTreeRegressor(random_state=42)
            # out_layer.fit(v, O)
        except:
            out_layer = np.linalg.lstsq(np.hstack([v, np.ones([v.shape[0], 1])]), O, rcond=None)[0]
        centers = I[used]
        ibias = k
        model_temp['centroids'] = I[used]
        model_temp['W'] = out_layer
        model_temp['Radius'] = k

        pred_val = self._predict(model_temp, X_val)
        sse_val, rms_val, mae_val, mse_val = self.compute_metrics(pred_val, y_val)
        metrics = np.array([mae_val, sse_val])

        flag, obj_old, obj_max, obj_min = self._distance(metrics, obj_old, obj_max, obj_min)
        if not flag:
            remains += 1
        else:
            remains = 0
            centers = I[used]
            ibias = k
        while remains < self.keep and P.shape[1] > 0:
            wj = W[:, -1:]
            a = np.matmul(wj.T, P) / np.matmul(wj.T, wj)
            P = P - wj * a
            not_zero = np.ones((P.shape[1])) * np.finfo(np.float64).eps
            e = ((np.matmul(P.T, O) ** 2.) / np.matmul((P * P).sum(0)[:, np.newaxis], D[np.newaxis, :])).sum(1)
            next = e.argmax()
            W = np.append(W, P[:, next, np.newaxis], axis=1)
            used = np.append(used, idx[next])
            P = np.delete(P, next, 1)
            idx = np.delete(idx, next)
            v = (np.atleast_2d(I)[:, np.newaxis] - I[used][np.newaxis, :]) * ibias
            v = np.sqrt((v ** 2.).sum(-1))
            v = np.exp(-(v ** 3.))
            try:
                out_layer = np.linalg.lstsq(np.hstack([v, np.ones([v.shape[0], 1])]), O, rcond=None)[0]
            except:
                out_layer = np.linalg.lstsq(np.hstack([v, np.ones([v.shape[0], 1])]), O, rcond=None)[0]
            model_temp['centroids'] = I[used]
            model_temp['W'] = out_layer
            model_temp['Radius'] = k
            pred_val = self._predict(model_temp, X_val)
            sse_val, rms_val, mae_val, mse_val = self.compute_metrics(pred_val, y_val)
            metrics = np.array([mae_val, sse_val])

            flag, obj_old, obj_max, obj_min = self._distance(metrics, obj_old, obj_max, obj_min)
            if not flag:
                remains += 1
            else:
                remains = 0
                centers = I[used]
                ibias = k
        if centers.shape[0] < self.keep:
            centers = I[used]
            ibias = k
        model['centroids'] = centers
        model['Radius'] = ibias

        v = (np.atleast_2d(I)[:, np.newaxis] - model['centroids'][np.newaxis, :]) * model['Radius']
        v = np.sqrt((v ** 2.).sum(-1))
        v = np.exp(-(v ** 3.))
        try:
            out_layer = np.linalg.lstsq(np.hstack([v, np.ones([v.shape[0], 1])]), O, rcond=None)[0]
        except:
            return np.inf
        model['W'] = out_layer
        pred_val = self._predict(model, X_val)
        sse_val, rms_val, mae_val, mse_val = self.compute_metrics(pred_val, y_val)
        self.metrics = np.array([mae_val, sse_val])
        self.model = model
        self.ncenters = model['centroids'].shape[0]
        self.err = mae_val
        return sse_val
