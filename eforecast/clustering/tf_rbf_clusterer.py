import os
import pickle
import shutil
import warnings

import joblib
import pandas as pd

try:
    import torch
    from eforecast.deep_models.pytorch_2x.network import DeepNetwork

    print('Backend is set pytorch')
except:
    try:
        import tensorflow as tf
        from eforecast.deep_models.tf_2x.network import DeepNetwork

        print('Backend is set Tensorflow 2.10')
    except:
        raise ImportError('Cannot find backend')

warnings.filterwarnings("ignore", category=FutureWarning)


class TfRBFClusterer:

    def __init__(self, static_data, train=False, online=False, refit=False):
        self.fuzzy_models = None
        self.rule_names = None
        self.is_trained = False
        self.refit = refit
        self.static_data = static_data
        self.rated = static_data['rated']
        self.n_jobs = self.static_data['clustering']['n_jobs']
        self.var_fuzz = self.static_data['clustering']['rbf_var_imp']
        self.n_var_lin = self.static_data['clustering']['n_var_lin']
        self.min_samples = self.static_data['clustering']['min_samples']
        self.max_samples_ratio = self.static_data['clustering']['max_samples_ratio']
        self.experiment_tag = self.static_data['clustering']['params']['experiment_tag']
        self.thres_act = self.static_data['clustering']['thres_act']

        self.params = self.static_data['clustering']['params']
        self.params['experiment'] = self.static_data['experiments'][self.experiment_tag]
        self.params['data_types'] = {'row_all': self.static_data['clustering']['data_type']}
        self.params['n_rules'] = self.static_data['clustering']['n_rules']
        self.path_fuzzy = os.path.join(static_data['path_model'], 'cluster_organizer', 'RBF')
        try:
            if not self.refit:
                self.load()
        except:
            pass
        self.train = train
        self.online = online
        self.path_fuzzy = os.path.join(static_data['path_model'], 'cluster_organizer', 'RBF')
        if not os.path.exists(self.path_fuzzy):
            os.makedirs(self.path_fuzzy)


    def transfer_learning_source(self):
        fuzzy_file = os.path.join(self.path_fuzzy, 'fuzzy_model.pickle')
        paths = []
        for i, fuzzy_model in enumerate(self.fuzzy_models):
            paths.append(os.path.join(self.path_fuzzy, f'RBF_fuzzy_net_{i}'))
        return paths, fuzzy_file

    def transfer_learning_target(self, paths, fuzzy_file):
        shutil.copy(fuzzy_file, os.path.join(self.path_fuzzy, 'fuzzy_model.pickle'))
        for i, path in enumerate(paths):
            shutil.copytree(path, os.path.join(self.path_fuzzy, f'RBF_fuzzy_net_{i}'), dirs_exist_ok=True)

    def compute_activations(self, with_predictions=False):
        if not hasattr(self, 'fuzzy_models'):
            raise ValueError('clusterer is not trained, fuzzy_models is not exists')
        activations = None
        predictions = None
        for i, fuzzy_model in enumerate(self.fuzzy_models):
            path_fuzzy_net = os.path.join(self.path_fuzzy, f'RBF_fuzzy_net_{i}')
            network = DeepNetwork(self.static_data, path_fuzzy_net, online=self.online, train=self.train)
            y_pred, act = network.predict(with_activations=True)
            activations = act if activations is None else pd.concat([activations, act], axis=1)
            predictions = y_pred if predictions is None else pd.concat([predictions, y_pred], axis=1)

        activations.columns = self.rule_names
        if self.static_data['horizon_type'] == 'multi-output':
            cols = [f'rbf_clusterer_{i}_{tag}' for tag in self.static_data['target_variable']['columns']
                    for i in range(len(self.fuzzy_models))]
        else:
            cols = [f'rbf_clusterer_{i}' for i in range(len(self.fuzzy_models))]
        predictions.columns = cols
        if with_predictions:
            return predictions, activations
        else:
            return activations

    def fit(self, cv_masks):
        if not self.refit and self.is_trained:
            return
        fuzzy_models = []
        activations = None
        calendar_vars = {'month': 1 / 12, 'day': 1 / 30, 'dayweek': 1 / 7, 'hour': 1 / 24, 'minute': 1 / 60}
        for n_case, var_imp in enumerate(self.var_fuzz):
            self.params['name'] = 'RBF_clustering'
            self.params['var_imp'] = []
            for x in var_imp:
                if x in calendar_vars.keys():
                    self.params['var_imp'].extend([f'{x}_sin', f'{x}_cos'])
                else:
                    self.params['var_imp'].append(x)
            self.params['min_samples'] = self.min_samples
            self.params['max_samples_ratio'] = self.max_samples_ratio
            self.params['thres_act'] = self.thres_act
            self.params['method'] = 'Fuzzy-MLP'
            path_fuzzy_net = os.path.join(self.path_fuzzy, f'RBF_fuzzy_net_{n_case}')
            if not os.path.exists(path_fuzzy_net):
                os.makedirs(path_fuzzy_net)
            network = DeepNetwork(self.static_data, path_fuzzy_net, self.params, is_global=True, is_fuzzy=True,
                                  online=self.online, train=self.train, refit=self.refit)
            network.fit(cv_masks, gpu_id='cpu')
            y_pred, act = network.predict(with_activations=True)
            activations = act if activations is None else pd.concat([activations, act], axis=1)
            fuzzy_models.append({'var_imp': var_imp})

        self.rule_names = ['rule_' + str(i) for i in range(activations.shape[1])]
        self.fuzzy_models = fuzzy_models
        self.is_trained = True
        self.save()

    def load(self):
        if os.path.exists(os.path.join(self.path_fuzzy, 'fuzzy_model.pickle')):
            try:
                f = open(os.path.join(self.path_fuzzy, 'fuzzy_model.pickle'), 'rb')
                tmp_dict = joblib.load(f)
                f.close()
                self.__dict__.update(tmp_dict)
            except:
                raise ImportError('Cannot open fuzzy model')
        else:
            raise ImportError('Cannot find fuzzy model')

    def save(self):
        f = os.path.join(self.path_fuzzy, 'fuzzy_model.pickle')
        dict = {}
        for k in self.__dict__.keys():
            if k not in ['static_data', 'refit', 'path_fuzzy']:
                dict[k] = self.__dict__[k]
        joblib.dump(dict, f)
