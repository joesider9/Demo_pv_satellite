import os
import pickle

from eforecast.common_utils.dataset_utils import sync_data_with_dates
from eforecast.common_utils.dataset_utils import sync_datasets

from eforecast.deep_models.tf_2x.network import DeepNetwork
from eforecast.clustering.clustering_manager import ClusterOrganizer
from eforecast.datasets.data_feeder import DataFeeder


class DistributedDeepNetwork:

    def __init__(self, static_data, path_network, params=None, is_online=False, train=False, refit=False):
        self.results = None
        self.best_mae_test = None
        self.is_trained = False
        self.is_online = is_online
        self.train = train
        self.refit = refit
        self.static_data = static_data
        if params is not None:
            self.params = params
            self.experiment_tag = self.params['experiment_tag']
            self.scale_method = self.params['data_type']['scaling']
            self.merge = self.params['data_type']['merge']
            self.compress = self.params['data_type']['compress']
            self.what_data = self.params['data_type']['what_data']
            self.thres_act = self.params['thres_act']
            self.rbf_var_imp = self.params['rbf_var_imp']
            self.min_samples = self.params['min_samples']
            self.max_samples_ratio = self.params['max_samples_ratio']
            self.n_rules = self.params['n_rules']
            self.is_fuzzy = self.params['is_fuzzy']
            self.clustering_method = self.params['clustering_method']
            if self.is_fuzzy and self.clustering_method is not None:
                self.clustering_method = None
                # raise ValueError('If is_Fuzzy is set True clustering method should be None')
        self.path_network = path_network
        try:
            if not self.refit:
                self.load()
        except:
            pass
        self.is_online = is_online
        self.train = train
        if not hasattr(self, 'params'):
            raise RuntimeError('The network has no params. You should train the net providing params')

    def feed_data(self):
        print('Read data for Clustering....')
        data_feeder = DataFeeder(self.static_data, online=self.is_online, train=self.train)
        X_imp, metadata_imp = data_feeder.feed_inputs(merge=self.merge, compress=self.compress,
                                                      scale_nwp_method=self.scale_method,
                                                      data_tag=self.what_data)
        var_imp = self.rbf_var_imp

        for var_name in var_imp:
            var_names = [c for c in X_imp.columns if var_name.lower() in c.lower()]
            if var_name not in X_imp.columns:
                if len(var_names) == 0:
                    raise ValueError(f'Cannot find variables associated with {var_name}')
                X_imp[var_name] = X_imp.loc[:, var_names].mean(axis=1)

        return X_imp[var_imp]

    def predict(self, x, metadata):
        X_imp, activations = None, None
        if self.clustering_method is None:
            X_imp = self.feed_data()
        else:
            cluster_organizer = ClusterOrganizer(self.static_data, is_online=self.is_online, train=self.train,
                                                 refit=False)
            activations, _ = cluster_organizer.compute_activations(self.clustering_method)
        network = DeepNetwork(self.static_data, self.path_network, params=self.params, is_global=True, is_fuzzy=True,
                              refit=False)
        predictions = network.predict(x, metadata, X_imp=X_imp, activations=activations)
        return predictions

    def fit(self, x, y, cv_mask, metadata, gpu_id=0):
        if not self.refit and self.is_trained:
            return self.best_mae_test
        X_imp, activations = None, None
        if self.clustering_method is None:
            X_imp = self.feed_data()
        else:
            cluster_organizer = ClusterOrganizer(self.static_data, is_online=self.is_online, train=self.train,
                                                 refit=False)
            activations, _ = cluster_organizer.compute_activations(self.clustering_method)

        self.params['var_imp'] = self.rbf_var_imp
        self.params['thres_act'] = self.thres_act
        self.params['min_samples'] = self.min_samples
        self.params['max_samples_ratio'] = self.max_samples_ratio
        self.params['groups'] = metadata['groups']
        self.params['method'] = f'Distributed-{self.experiment_tag.upper()}'
        network = DeepNetwork(self.static_data, self.path_network, params=self.params, is_global=True,
                              is_fuzzy=self.is_fuzzy, refit=self.refit)

        network.fit(x, y, cv_mask, metadata, X_imp=X_imp, activations=activations, gpu_id=gpu_id)

        self.best_mae_test = network.best_mae_test
        self.best_mae_val = network.best_mae_val
        self.best_sse_test = network.best_sse_test
        self.best_sse_val = network.best_sse_val
        self.results = network.results

        self.is_trained = True
        self.save()

    def load(self):
        if os.path.exists(os.path.join(self.path_network, 'distributed_model.pickle')):
            try:
                f = open(os.path.join(self.path_network,
                                      'distributed_model.pickle'), 'rb')
                tmp_dict = pickle.load(f)
                f.close()
                self.__dict__.update(tmp_dict)
            except:
                raise ImportError('Cannot open distributed_model.pickle')
        else:
            raise ImportError('Cannot find distributed_model')

    def save(self):
        f = open(os.path.join(self.path_network, 'distributed_model.pickle'), 'wb')
        dict_self = {}
        for k in self.__dict__.keys():
            if k not in ['logger', 'static_data', 'path_network', 'refit', 'is_online', 'train']:
                dict_self[k] = self.__dict__[k]
        pickle.dump(dict_self, f)
        f.close()
