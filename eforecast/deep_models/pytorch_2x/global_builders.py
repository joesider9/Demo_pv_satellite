import copy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from eforecast.common_utils.train_utils import distance
from eforecast.deep_models.pytorch_2x.trainer import compute_tensors
from eforecast.deep_models.pytorch_2x.trainer import feed_data_eval
from eforecast.deep_models.pytorch_2x.trainer import evaluate_activations

from eforecast.common_utils.train_utils import linear_output


def check_rbf_bounds(model, N, X_train, y_train, X_val, y_val, X_test, y_test,
                     params, best_clusters, device):
    warm = 0
    if best_clusters is None:
        raise ValueError('best_clusters is not computed')
    act_train = compute_tensors(model, feed_data_eval(X_train)).cpu().detach().numpy()
    act_val = compute_tensors(model, feed_data_eval(X_val)).cpu().detach().numpy()
    act_test = compute_tensors(model, feed_data_eval(X_test)).cpu().detach().numpy()
    mae_val_lin, mae_test_lin, sse_val_lin, sse_test_lin = linear_output(X_train, X_val, X_test,
                                                                         y_train, y_val, y_test,
                                                                         act_train, act_val, act_test,
                                                                         params)
    sum_act, min_act, max_act, mean_act, id_min, id_max = evaluate_activations(model,
                                                                               X_train,
                                                                               params['thres_act'])
    sum_act, min_act, max_act, mean_act, id_min, id_max = (sum_act.cpu().detach().numpy(),
                                                           min_act.cpu().detach().numpy(),
                                                           max_act.cpu().detach().numpy(),
                                                           mean_act.cpu().detach().numpy(),
                                                           id_min.cpu().detach().numpy(),
                                                           id_max.cpu().detach().numpy())
    min_samples = params['min_samples']
    max_samples = int(params['max_samples_ratio'] * y_train.shape[0])
    if min_act < min_samples:
        assign_rbf(model, best_clusters, device, idx=id_min)
    if max_act > max_samples:
        assign_rbf(model, best_clusters, device, idx=id_max)

    return mae_val_lin, mae_test_lin, sse_val_lin, sse_test_lin, sum_act, min_act, max_act, mean_act, warm


def assign_rbf(model, best_clusters, device, idx=None):
    with torch.no_grad():
        for name, variable in model.named_parameters():
            if idx is None:
                if f'RBF_variance' in name:
                    variable.data = torch.from_numpy(best_clusters[name]).to(device)
                    variable.requires_grad = True
            else:
                if f'RBF_variance_{idx}' in name:
                    variable.data = torch.from_numpy(best_clusters[name]).to(device)
                    variable.requires_grad = True
        print('Assign new values')


def get_rbf(model):
    with torch.no_grad():
        best_clusters = dict()
        for name, variable in model.named_parameters():
            if 'centroid' in name or 'RBF_variance' in name:
                best_clusters[name] = variable.data.cpu().numpy()
    return best_clusters


class RBFLayer(nn.Module):
    def __init__(self, params):
        super(RBFLayer, self).__init__()
        self.var = nn.ParameterDict()
        self.centroids = nn.ParameterDict()
        self.thres_act = params['thres_act']
        centroids = params['centroids']
        var_init = params['var_init']
        self.n_rules = centroids.shape[0]
        self.n_var = centroids.shape[1]
        self.var_init = []
        self.centroids_init = []
        for n in range(self.n_rules):
            self.var_init.append(torch.Tensor(var_init.iloc[n].values.reshape(1, -1)))
            self.centroids_init.append(torch.Tensor(centroids[n].reshape(1, -1)))

        for n in range(self.n_rules):
            self.centroids[f'centroid_{n}'] = nn.Parameter(self.centroids_init[n], requires_grad=False)
            self.var[f'RBF_variance_{n}'] = nn.Parameter(self.var_init[n], requires_grad=True)

    def forward(self, inputs):
        s = inputs.size()
        phi = torch.Tensor([]).to(inputs.device)
        for n in range(self.n_rules):
            if self.centroids[f'centroid_{n}'].device != inputs.device:
                self.centroids[f'centroid_{n}'] = self.centroids[f'centroid_{n}'].to(inputs.device)
            if self.var[f'RBF_variance_{n}'].device != inputs.device:
                self.var[f'RBF_variance_{n}'] = self.var[f'RBF_variance_{n}'].to(inputs.device)
            d1 = torch.abs(inputs - torch.tile(self.centroids[f'centroid_{n}'], (s[0], 1)))
            sqrd = torch.sum(torch.pow(torch.div(d1, torch.tile(self.var[f'RBF_variance_{n}'],
                                                                        (s[0], 1))), 2), 1)
            d = torch.sqrt(sqrd + 1e-8)
            p = torch.unsqueeze(torch.exp(torch.mul(-1, torch.square(d))), 1)
            phi = torch.cat([phi, p], 1)
        return phi


class apply_activations(nn.Module):
    def __init__(self, thres_act):
        super(apply_activations, self).__init__()
        self.thres_act = thres_act

    def forward(self, x, act):
        output_shape = x.size()[-1]
        model_output = torch.mul(torch.tile(act, (1, output_shape)), x)
        return model_output


class act_nan_layer(nn.Module):
    def __init__(self, thres_act):
        super(act_nan_layer, self).__init__()
        self.thres_act = thres_act

    def forward(self, act_all, **kwargs):
        act_all = act_all - self.thres_act
        act_all = torch.ceil(act_all)
        act = torch.sum(act_all, -1)
        act_nan_err = act[act == 0]
        act_nan_err = act_nan_err.size()[0]

        return act_nan_err

def check_fuzzy_performance(net_model, N,
                            X_train, y_train,
                            X_val, y_val,
                            X_test, y_test,
                            params,
                            init_clusters, best_clusters,
                            device, mae_old_lin,
                            mae_max_lin, mae_min_lin,
                            sse_old_lin, sse_max_lin,
                            sse_min_lin, explode_clusters):
    mae_val_lin, mae_test_lin, sse_val_lin, sse_test_lin, \
        sum_act, min_act, max_act, mean_act, warm = \
        check_rbf_bounds(net_model, N,
                         X_train, y_train,
                         X_val, y_val,
                         X_test, y_test,
                         params, init_clusters, device)
    mae_lin = np.hstack([mae_val_lin, mae_test_lin])
    flag_mae_lin, mae_old_lin, mae_max_lin, mae_min_lin = distance(mae_lin, mae_old_lin,
                                                                   mae_max_lin, mae_min_lin)
    sse_lin = np.hstack([sse_val_lin, sse_test_lin])
    flag_sse_lin, sse_old_lin, sse_max_lin, sse_min_lin = distance(sse_lin, sse_old_lin,
                                                                   sse_max_lin, sse_min_lin)
    if (flag_mae_lin and flag_sse_lin) and not (mae_val_lin > 1 or mae_test_lin > 1):
        best_clusters = get_rbf(net_model)
    if (mae_val_lin > 1 or mae_test_lin > 1) \
            and explode_clusters:
        for param, weight in best_clusters.items():
            if 'RBF_variance' in param:
                weight *= 1.25
        assign_rbf(net_model, best_clusters, device)
        mae_val_lin, mae_test_lin, sse_val_lin, sse_test_lin, \
            sum_act, min_act, max_act, mean_act, warm = \
            check_rbf_bounds(net_model, N,
                             X_train, y_train,
                             X_val, y_val,
                             X_test, y_test,
                             params, init_clusters, device)
        warm = 4
    return (net_model, best_clusters, sum_act, min_act, max_act, mean_act, warm, mae_old_lin, mae_max_lin,
            mae_min_lin, sse_old_lin, sse_max_lin, sse_min_lin, mae_val_lin, mae_test_lin)
