import os
import shutil

import joblib
import numpy as np
import pandas as pd

from eforecast.common_utils.clustering_utils import check_if_all_nans
from eforecast.clustering.tf_rbf_clusterer import TfRBFClusterer

from eforecast.datasets.data_preprocessing.data_split import Splitter
from eforecast.datasets.files_manager import FilesManager


class ClusterOrganizer:

    def __init__(self, static_data, is_online=False, train=False, refit=False):
        self.is_online = is_online
        self.train = train
        self.refit = refit
        self.sampled_data = None
        self.static_data = static_data
        self.path_model = self.static_data['path_model']
        self.is_Fuzzy = self.static_data['is_Fuzzy']
        self.thres_act = self.static_data['clustering']['thres_act']
        self.scale_row_method = self.static_data['clustering']['data_type']['scale_row_method']
        self.scale_nwp_method = self.static_data['clustering']['data_type']['scale_nwp_method']
        self.merge = self.static_data['clustering']['data_type']['merge']
        self.compress = self.static_data['clustering']['data_type']['compress']
        self.methods = self.static_data['clustering']['methods']
        self.make_clusters_for_method = self.static_data['clustering']['clusters_for_method']
        self.file_manager = FilesManager(static_data, is_online=is_online, train=train)


    @staticmethod
    def create_cluster_folders(path):
        if os.path.exists(path):
            shutil.rmtree(path)
        if not os.path.exists(path):
            os.makedirs(path)

    def get_fuzzy_models(self):
        for method in self.methods:
            if method == 'RBF':
                clusterer = TfRBFClusterer(self.static_data, train=self.train, online=self.is_online)
            else:
                raise NotImplementedError(
                    f"Clustering method {self.static_data['clustering']['method']} not implemented")

            return clusterer.transfer_learning_source()

    def copy_fuzzy_models(self, paths, fuzzy_file):
        for method in self.methods:
            if method == 'RBF':
                clusterer = TfRBFClusterer(self.static_data, train=self.train, online=self.is_online)
            else:
                raise NotImplementedError(
                    f"Clustering method {self.static_data['clustering']['method']} not implemented")

            return clusterer.transfer_learning_target(paths, fuzzy_file)

    def fit(self):
        if self.is_Fuzzy:
            cv_mask = self.file_manager.check_if_exists_cv_data()
            for method in self.methods:
                if method == 'RBF':
                    clusterer = TfRBFClusterer(self.static_data, train=self.train, online=self.is_online,
                                               refit=self.refit)
                else:
                    raise NotImplementedError(
                        f"Clustering method {self.static_data['clustering']['method']} not implemented")
                if not clusterer.is_trained or self.refit:
                    clusterer.fit(cv_mask)

    def cluster_input_dates(self, clusterer):
        clustered_dates = dict()
        activations = clusterer.compute_activations()
        activations = check_if_all_nans(activations, self.thres_act)
        for cluster in clusterer.rule_names:
            indices = np.where(activations[cluster] > self.thres_act)[0]
            clustered_dates[cluster] = activations.index[indices]
        return clustered_dates, activations

    def predict(self, method):
        if method not in self.methods:
            raise ValueError(f'{method} is not in clustering methods')
        if method == 'RBF':
            clusterer = TfRBFClusterer(self.static_data, train=self.train, online=self.is_online)
        else:
            raise NotImplementedError(
                f"Clustering method {self.static_data['clustering']['method']} not implemented")

        predictions, activations = clusterer.compute_activations(with_predictions=True)
        return predictions, activations

    def compute_activations(self, method):
        if method not in self.methods:
            raise ValueError(f'{method} is not in clustering methods')
        if method == 'RBF':
            clusterer = TfRBFClusterer(self.static_data, train=self.train, online=self.is_online)
        else:
            raise NotImplementedError(
                f"Clustering method {self.static_data['clustering']['method']} not implemented")
        clustered_dates, activations = self.cluster_input_dates(clusterer)
        return activations, clustered_dates

    def create_clusters_and_cvs(self):
        cv_mask = self.file_manager.check_if_exists_cv_data()
        path_global = os.path.join(self.static_data['path_model'], 'global')
        if not os.path.exists(path_global):
            os.makedirs(path_global)
        joblib.dump(cv_mask, os.path.join(path_global, 'cv_mask.pickle'))
        if self.is_Fuzzy:
            clusters = dict()
            for method in self.methods:
                if method == 'RBF':
                    clusterer = TfRBFClusterer(self.static_data, train=self.train, online=self.is_online)
                else:
                    raise NotImplementedError(
                        f"Clustering method {self.static_data['clustering']['method']} not implemented")
                if method == self.make_clusters_for_method or self.make_clusters_for_method == 'both':
                    clustered_dates, activations = self.cluster_input_dates(clusterer)
                    splitter = Splitter(self.static_data, is_online=self.is_online, train=self.train)
                    cluster_cvs = splitter.split_cluster_data(clustered_dates, cv_mask=cv_mask)
                    cv_mask = cluster_cvs['global']
                    self.file_manager.remove_cv_data_files()
                    self.file_manager.save_cv_data(cv_mask)
                    joblib.dump(cv_mask, os.path.join(path_global, 'cv_mask.pickle'))
                    for cluster in clusterer.rule_names:
                        path = os.path.join(clusterer.path_fuzzy, cluster)
                        if os.path.exists(os.path.join(path, 'cv_mask.pickle')) and not self.refit:
                            clusters[f'{method}_{cluster}'] = path
                            continue
                        self.create_cluster_folders(path)
                        mask_train, mask_val, mask_test = cluster_cvs[cluster]
                        print(f'Cluster {cluster} {mask_train.shape[0]}, train samples {mask_val.shape[0]} val samples, '
                              f'{mask_test.shape[0]} test samples')
                        joblib.dump([mask_train, mask_val, mask_test], os.path.join(path, 'cv_mask.pickle'))
                        clusters[f'{method}_{cluster}'] = path
            joblib.dump(clusters, os.path.join(self.path_model, 'clusters.pickle'))

    def update_cluster_folders(self):
        if self.is_Fuzzy:
            clusters = dict()
            for method in self.methods:
                if method == 'RBF':
                    clusterer = TfRBFClusterer(self.static_data, train=self.train, online=self.is_online)
                else:
                    raise NotImplementedError(
                        f"Clustering method {self.static_data['clustering']['method']} not implemented")
                if method == self.make_clusters_for_method or self.make_clusters_for_method == 'both':
                    for cluster in clusterer.rule_names:
                        path = os.path.join(clusterer.path_fuzzy, cluster)
                        if not os.path.exists(path):
                            raise ImportError(f'Cannot find {cluster} path: {path}')
                        clusters[f'{method}_{cluster}'] = path
            joblib.dump(clusters, os.path.join(self.path_model, 'clusters.pickle'))
