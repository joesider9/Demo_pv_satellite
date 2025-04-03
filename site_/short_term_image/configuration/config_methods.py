import numpy as np
import pandas as pd
import os

import timm

from site_.short_term_image.configuration.config_utils import define_n_jobs
from site_.short_term_image.configuration.config_utils import find_pycharm_path
from site_.short_term_image.configuration.config_utils import define_enviroment
from site_.short_term_image.configuration.config_project import TYPE
from site_.short_term_image.configuration.config_project import HORIZON_TYPE
from site_.short_term_image.configuration.config_input_data import NWP_DATA_MERGE
from site_.short_term_image.configuration.config_input_data import DATA_COMPRESS
from site_.short_term_image.configuration.config_input_data import DATA_ROW_SCALE
from site_.short_term_image.configuration.config_input_data import DATA_NWP_SCALE
from site_.short_term_image.configuration.experiments import experiments

RUNTIME_BACKEND = 'TORCH'  # TF_2, TORCH

from site_.short_term_image.configuration.timm_model_list import timm_models
from site_.short_term_image.configuration.huggingface_vit_list import huggingface_models

vit_models = timm_models.union(huggingface_models)

Transformer_models = {'Autoformer', 'Crossformer', 'DLinear', 'FEDformer', 'FiLM', 'Informer', 'LightTS', 'PatchTST',
                      'Reformer'}
ENV_NAME, PATH_ENV = define_enviroment(RUNTIME_BACKEND)

METHODS_CPU = {'RF': False,
               'CatBoost': False,
               'lasso': False,
               'RBFols': False,
               'GA_RBFols': False,
               }

CLUSTER_METHODS = {'CNN': False,
                   'TIMM': True,
                   'LSTM': False,
                   'TRANSFORMER': False,
                   'MLP': False,
                   }

GLOBAL_METHODS = {'CrossViVit': False,
                  'CNN': False,
                  'TIMM': True,
                  'TRANSFORMER': False,
                  'Distributed': False,
                  'LSTM': False,
                  }

TRANSFER_LEARNING = False
TRANSFER_LEARNING_FROM = {'configuration': '/path/to/model'}
TRANSFER_LEARNING_MERGE_DATA = True


FEATURE_SELECTION_METHODS = [None, 'Leshy_pimp', 'Leshy_shap']
                                             # Could be None, 'Leshy_native','Leshy_pimp','Leshy_shap',
                                             # 'BoostAGroota_native','BoostAGroota_pimp','BoostAGroota_shap',

combine_methods = ['bcp', 'kmeans']  # 'bcp', 'elastic_net', 'kmeans', 'CatBoost'

ACCEPTABLE_PERFORMANCE = 2

DATA_TYPE_TO_CLUSTER_COMBINE = {'scale_row_method': DATA_ROW_SCALE[0],
                                'scale_nwp_method': DATA_NWP_SCALE[0],
                                'merge': NWP_DATA_MERGE[0],
                                'compress': DATA_COMPRESS[0]
                                }


def rbf_variables():
    if TYPE == 'pv':
        var_imp = [['hour', 'Flux']]
    elif TYPE == 'wind':
        var_imp = [['direction', 'wind']]
    elif TYPE == 'load':
        var_imp = [['sp_index', 'hour', 'month', 'Temp']]
    elif TYPE == 'FA':
        var_imp = [['sp_index', 'month', 'temp_max', 'dayweek']]
    else:
        var_imp = [[]]
    return var_imp


def config_methods():
    static_data = dict()
    static_data['env_name'] = ENV_NAME
    static_data['path_env'] = PATH_ENV
    static_data['backend'] = RUNTIME_BACKEND
    static_data['methods_cpu'] = METHODS_CPU
    static_data['cluster_methods'] = CLUSTER_METHODS
    static_data['global_methods'] = GLOBAL_METHODS
    static_data['transfer_learning'] = TRANSFER_LEARNING
    static_data['transfer_learning_from'] = TRANSFER_LEARNING_FROM
    static_data['transfer_learning_merge_data'] = TRANSFER_LEARNING_MERGE_DATA

    
    static_data['base_data_types'] = DATA_TYPE_TO_CLUSTER_COMBINE
    static_data['feature_selection_methods'] = FEATURE_SELECTION_METHODS
    static_data['experiments'] = experiments
    static_data['max_performance'] = ACCEPTABLE_PERFORMANCE
    static_data['val_test_ratio'] = 0.125
    static_data['CVs'] = 2

    n_jobs = define_n_jobs()
    static_data['n_jobs'] = n_jobs['n_jobs']
    static_data['n_gpus'] = n_jobs['n_gpus']
    static_data['intra_op'] = n_jobs['intra_op']
    static_data['kernels'] = {'nwp': [[2, 2, 1], [2, 1, 2], [2, 2, 2], [1, 1, 2], [1, 2, 1], [1, 2, 2]],
                              'images': [[2, 5, 5], [2, 10, 10], [2, 40, 40], [4, 30, 40], [4, 30, 30], [2, 30, 30]]}
    static_data['CNN'] = {'experiment_tag': {'cnn1', 'cnn2', 'cnn3'},
                          'n_trials': 12,
                          'filters': [24, 48],
                          'stride': 1,
                          'conv_dim': 2,
                          'batch_norm': {True, False},
                          'area_adjust': 24,
                          'final_image_size': 64,
                          'act_func': 'elu',
                          'sat_image_type': 'RBG:Infrared',
                          'create_image_dataset_real_time': True,
                          'merge': set(NWP_DATA_MERGE),
                          'compress': None,
                          'scale_row_method': None,
                          'scale_nwp_method': set(DATA_NWP_SCALE),
                          'feature_selection_method': None,
                          'optimizer': 'adam',
                          'scheduler': {'CosineAnnealing'},
                          'transfer_learning_from_global': True,
                          'learning_rate': 1e-4,
                          'batch_size': 128,
                          'max_iterations': 600,
                          'warming_iterations': 0,
                          'n_jobs': 3
                          }
    static_data['CrossViVit'] = {'experiment_tag': {'CrossViVit_net'},
                                 'n_trials': 2,
                                 'model_name': 'visformer_small',
                                 'timm_models': ' '.join(timm_models),
                                 'transformer_name': None,
                                 'sat_image_type': {#'RBG:Infrared1:Infrared2:coord',
                                                    'RBG:Infrared1:RBG_opt',
                                                    #'RBG_grey:Infrared1:Infrared2:RBG_opt'
                                 },
                                 'pretrained': {'tune'},
                                 'transfer_learning_from_global': False,
                                 'area_adjust': 24,
                                 'final_image_size': {64, 128, 224},
                                 'create_image_dataset_real_time': True,
                                 'stride': 1,
                                 'filters': 12,
                                 'conv_dim': 3,
                                 'optimizer': 'adamw',
                                 'scheduler': {'CosineAnnealing'},
                                 'merge': NWP_DATA_MERGE[0],
                                 'compress': DATA_COMPRESS[0],
                                 'scale_row_method': DATA_ROW_SCALE[0],
                                 'scale_nwp_method': DATA_NWP_SCALE[0],
                                 'feature_selection_method': None,
                                 'use_data': 3,
                                 # 1. all, 2. image and calendar 3. calendar and meas, 4. image
                                 'act_func': 'gelu',
                                 'batch_size': 64,
                                 'max_iterations': 30,
                                 'warming_iterations': 0,
                                 'learning_rate': {1e-4, 1e-5},
                                 'n_jobs': 1
                                 }
    static_data['TRANSFORMER'] = {'experiment_tag': {'transformer'},
                                  'n_trials': 8 * len(Transformer_models),
                                  'transformer_name': Transformer_models,
                                  'transfer_learning_from_global': False,
                                  'merge': None,
                                  'compress': None,
                                  'scale_row_method': set(DATA_ROW_SCALE),
                                  'scale_nwp_method': None,
                                  'feature_selection_method': set(FEATURE_SELECTION_METHODS),
                                  'act_func': 'gelu',
                                  'optimizer': 'adam',
                                  'scheduler': {'CosineAnnealing'},
                                  'batch_size': {64, 128},
                                  'max_iterations': 600,
                                  'warming_iterations': 0,
                                  'learning_rate': {5e-5, 1e-4, 0.5e-3},
                                  'n_jobs': 3
                                  }
    static_data['TIMM'] = {'experiment_tag': {'timm_net'},
                           'n_trials':len(vit_models),
                           'model_name': vit_models,
                           'timm_models': ' '.join(timm_models),
                           'pretrained': {'tune'},  # 'initial', 'tune', 'finetune' better 'tune
                           'sat_image_type': 'RBG:Infrared',
                           'final_image_size': 128,
                           'transfer_learning_from_global': True,
                           'optimizer': 'adam',
                           'scheduler': {'CosineAnnealing'},
                           'fix_grid_search': True,
                           'merge': None,
                           'compress': None,
                           'scale_row_method': None,
                           'scale_nwp_method': None,
                           'feature_selection_method': None,
                           'create_image_dataset_real_time': True,
                           'temporal_lags': 0,
                           'area_adjust': 24,
                           'use_classes': False,  # better false
                           'filters': 12,
                           'conv_dim': 2,
                           'batch_size': 64,
                           'max_iterations': 200,
                           'warming_iterations': 0,
                           'learning_rate': 0.8e-4,
                           'use_data': 3,
                           # 1. all, 2. image and calendar 3. calendar and meas, 4. image
                           'act_func': 'gelu',
                           'n_jobs': 1
                           }
    static_data['LSTM'] = {'units': 24,
                           'n_trials': 15,
                           'experiment_tag': {'lstm1', 'lstm2', 'lstm3'},
                           'batch_size': {64, 128},
                           'transfer_learning_from_global': True,
                           'use_embedding': {True, False},
                           'merge': None,
                           'compress': None,
                           'scale_row_method': set(DATA_ROW_SCALE),
                           'scale_nwp_method': None,
                           'feature_selection_method': set(FEATURE_SELECTION_METHODS),
                           'act_func': {'sigmoid', 'gelu'},
                           'max_iterations': 600,
                           'warming_iterations': 5,
                           'optimizer': {'adam', 'adamw'},
                           'scheduler': {'CosineAnnealing'},
                           'learning_rate': [1e-4, 1e-3],
                           'n_jobs': 3
                           }
    static_data['MLP'] = {'experiment_tag': {'mlp2', 'mlp3'},
                          'n_trials': 12,
                          'hold_prob': 1,
                          'batch_size': [32, 200],
                          'transfer_learning_from_global': True,
                          'merge': set(NWP_DATA_MERGE),
                          'compress': set(DATA_COMPRESS),
                          'scale_row_method': set(DATA_ROW_SCALE),
                          'scale_nwp_method': set(DATA_NWP_SCALE),
                          'feature_selection_method': set(FEATURE_SELECTION_METHODS),
                          'act_func': {'elu', 'sigmoid', 'gelu'},
                          'max_iterations': 600,
                          'warming_iterations': 5,
                          'optimizer': {'adam', 'adamw'},
                          'scheduler': {'CosineAnnealing'},
                          'learning_rate': {1e-4, 1e-3},
                          'n_jobs': 3,
                          }
    static_data['Distributed'] = {'experiment_tag': {'distributed_cnn1', 'distributed_mlp2'},
                                  'keep_n_models': 2,
                                  'n_trials': 10,
                                  'hold_prob': 1,
                                  'batch_size': {32, 64, 128},
                                  'filters': 6,
                                  'conv_dim': 2,
                                  'transfer_learning_from_global': True,
                                  'data_type': DATA_TYPE_TO_CLUSTER_COMBINE,
                                  'merge': set(NWP_DATA_MERGE),
                                  'compress': set(DATA_COMPRESS),
                                  'scale_row_method': set(DATA_ROW_SCALE),
                                  'scale_nwp_method': set(DATA_NWP_SCALE),
                                  'feature_selection_method': set(FEATURE_SELECTION_METHODS),
                                  'act_func': {'sigmoid'},
                                  'max_iterations': 800,
                                  'optimizer': {'adam', 'adamw'},
                                  'scheduler': {'CosineAnnealing'},
                                  'warming_iterations': 4,
                                  'learning_rate': {1e-4, 1e-3},
                                  'thres_act': 0.001,
                                  'min_samples': 200,
                                  'max_samples_ratio': 0.8,
                                  'train_schedule': 'simple',  # simple or complex
                                  'n_rules': {9},
                                  'is_fuzzy': {False},  # If True creates its own activations
                                  'clustering_method': {'RBF', None},  # None RBF
                                  'rbf_var_imp': rbf_variables()[0],
                                  # Data type used for clustering when it is Fuzzy
                                  'n_jobs': n_jobs['n_jobs_cnn_3d'],
                                  }
    CLUSTERING_METHOD = ['RBF']  # RBF
    static_data['clustering'] = {'n_jobs': n_jobs['n_jobs'],
                                 'data_type': DATA_TYPE_TO_CLUSTER_COMBINE,
                                 'methods': CLUSTERING_METHOD,
                                 'clusters_for_method': 'RBF',  # RBF
                                 'prediction_for_method': ['RBF'],  # RBF
                                 'thres_act': 0.001,
                                 'rbf_var_imp': rbf_variables(),
                                 'explode_clusters': True,
                                 'n_var_lin': 12,
                                 'min_samples': 500,
                                 'max_samples_ratio': 0.5,
                                 'warming_iterations': 4,
                                 'n_rules': 8,
                                 'params': {'experiment_tag': 'exp_fuzzy1',
                                            'n_trials': 1,
                                            'hold_prob': 1,
                                            'batch_size': 256,
                                            'filters': 12,
                                            'conv_dim': 2,
                                            'merge': NWP_DATA_MERGE[0],
                                            'compress': DATA_COMPRESS[0],
                                            'scale_row_method': DATA_ROW_SCALE[0],
                                            'scale_nwp_method': DATA_NWP_SCALE[0],
                                            'feature_selection_method': None,
                                            'train_schedule': 'simple',  # Simple or complex
                                            'optimizer': 'adam',
                                            'scheduler': 'CosineAnnealing',
                                            'act_func': None,
                                            'max_iterations': 1000,
                                            'warming_iterations': 100,
                                            'learning_rate': 0.25e-3,
                                            'n_jobs': 1,
                                            }
                                 }
    static_data['RF'] = {'n_trials': 40,
                         'max_depth': {1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 32, 36, 42, 48},
                         'max_features': {'sqrt', 'log2', 0.8, 0.7, 0.5, 0.4, 0.3, 0.2},
                         'min_samples_leaf': [2, 200],
                         'min_samples_split': [2, 100],
                         'max_samples': [0.3, 1],
                         'oob_score': True
                         }
    static_data['CatBoost'] = {'n_trials': 40,
                               'iterations': 1000,
                               'learning_rate': {0.01, 0.05, 0.1, 0.2},
                               'l2_leaf_reg': [2, 6],
                               "objective": {"RMSE", "MAE"} if HORIZON_TYPE != 'multi-output' else {'MultiRMSE'},
                               'min_data_in_leaf': [2, 100],
                               "colsample_bylevel": [0.6, 1],
                               "depth": [5, 9],
                               "boosting_type": {"Ordered", "Plain"},
                               "bootstrap_type": {"Bayesian", "Bernoulli", "MVS"},
                               "eval_metric": "MAE" if HORIZON_TYPE != 'multi-output' else 'MultiRMSE',
                               }
    static_data['lasso'] = {'n_trials': 100,
                            'max_iter': 150000,
                            'eps': {0.005, 0.01, 0.05, 0.1, 0.2, 0.5, 1, 2},
                            'fit_intercept': {True, False},
                            "selection": {"cyclic", "random"},
                            }
    static_data['RBFols'] = {'n_trials': 10,
                             'warming_iterations': 4,
                             'width': [0.01, 15],
                             'keep': [2, 6]
                             }
    static_data['GA_RBFols'] = {'n_trials': 1000,
                                'warming_iterations': 4,
                                'width': [0.01, 15],
                                'keep': [2, 6]
                                }
    static_data['combining'] = {'methods': combine_methods,
                                'data_type': DATA_TYPE_TO_CLUSTER_COMBINE,
                                }

    return static_data
