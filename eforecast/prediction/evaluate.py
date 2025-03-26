import copy
import os
import joblib
import numpy as np
import pandas as pd

from eforecast.common_utils.eval_utils import compute_metrics
from eforecast.common_utils.dataset_utils import sync_datasets
from eforecast.datasets.data_feeder import DataFeeder
from eforecast.datasets.data_preprocessing.data_scaling import Scaler
from eforecast.datasets.files_manager import FilesManager

CategoricalFeatures = ['hour', 'month', 'sp_index', 'dayweek']


class Evaluator:
    def __init__(self, static_data, train=True, refit=False):
        self.static_data = static_data
        self.refit = refit
        self.is_Fuzzy = self.static_data['is_Fuzzy']
        self.global_methods = [method for method, values in static_data['global_methods'].items() if values]
        if self.is_Fuzzy:
            self.clusters = joblib.load(os.path.join(static_data['path_model'], 'clusters.pickle'))

        self.predictions = joblib.load(os.path.join(self.static_data['path_data'],
                                                    'predictions_regressors_train.pickle'))

        self.predictions_eval = joblib.load(os.path.join(self.static_data['path_data'],
                                                         'predictions_regressors_eval.pickle'))
        self.scaler = Scaler(static_data, recreate=False, online=False, train=True)
        self.scale_target_method = self.static_data['scale_target_method']
        self.rated = self.static_data['rated']
        self.multi_output = True if self.static_data['horizon_type'] == 'multi-output' else False
        self.evaluator_path = os.path.join(self.static_data['path_model'], 'Results')
        if not os.path.exists(self.evaluator_path):
            os.makedirs(self.evaluator_path)

    def feed_target(self, train=False, inverse=False):
        print('Read target for evaluation....')
        data_feeder = DataFeeder(self.static_data, train=train)
        y = data_feeder.feed_target(inverse=inverse)
        return y

    def evaluate_methods_for_cluster(self, clusterer_method=None, cluster_name=None, trial=None):
        results_methods = pd.DataFrame()
        y_scaled = self.feed_target(train=True)
        y_eval_scaled = self.feed_target()
        if clusterer_method is not None:
            methods_predictions = self.predictions['clusters'][clusterer_method][cluster_name]
        else:
            methods_predictions = self.predictions['global']
        for method in methods_predictions.keys():
            pred_train_scaled = methods_predictions[method]
            if pred_train_scaled.shape[0] == 0:
                continue
            if clusterer_method is not None:
                pred_eval_scaled = self.predictions_eval['clusters'][clusterer_method][cluster_name][method]
            else:
                pred_eval_scaled = self.predictions_eval['global'][method]

            pred_train = self.scaler.inverse_transform_data(pred_train_scaled,
                                                            f'target_{self.scale_target_method}')
            if pred_eval_scaled.shape[0] != 0:
                pred_eval = self.scaler.inverse_transform_data(pred_eval_scaled,
                                                               f'target_{self.scale_target_method}')
            else:
                pred_eval = pred_eval_scaled
            if clusterer_method is not None:
                cv_masks = joblib.load(os.path.join(self.clusters[cluster_name], 'cv_mask.pickle'))
            else:
                cv_masks = joblib.load(os.path.join(self.static_data['path_model'], 'global', 'cv_mask.pickle'))
            cv_names = ['val', 'test']
            results_methods_temp = pd.DataFrame()
            if pred_eval_scaled.shape[0] != 0:
                cv_mask = pred_eval.index.intersection(y_eval_scaled.index)
                y_eval = self.scaler.inverse_transform_data(y_eval_scaled,
                                                            f'target_{self.scale_target_method}')
                res_eval = compute_metrics(pred_eval.loc[cv_mask], y_eval.loc[cv_mask], self.rated,
                                           f'{cluster_name}_{method}')['mae'].to_frame()
                res_eval.columns = [f'{column}_eval' for column in res_eval.columns]
                results_methods_temp = res_eval
            if self.static_data['horizon_type'] == 'multi-output':
                col = [f'hour_ahead_{i}' for i in range(self.static_data['horizon'])]
            else:
                col = y_scaled.columns[0]
            name_col = 'target'
            y = self.scaler.inverse_transform_data(y_scaled[col],
                                                   f'target_{self.scale_target_method}')
            p_scaled = copy.deepcopy(pred_train_scaled)
            p = copy.deepcopy(pred_train)
            for cv_name, cv_mask in zip(cv_names, cv_masks[1:]):
                cv_mask = cv_mask.intersection(p_scaled.index)

                res = compute_metrics(p.loc[cv_mask], y[col].loc[cv_mask],
                                      self.rated,
                                      f'{cluster_name}_{method}')['mae'].to_frame()
                res.columns = [f'{column}_{name_col}_{cv_name}'
                               for column in res.columns]
                results_methods_temp = pd.concat([results_methods_temp, res], axis=1)
            results_methods = pd.concat([results_methods, results_methods_temp])
        if 'mae_eval' in results_methods.columns:
            empty_row = results_methods.corrwith(results_methods['mae_eval']). \
                to_frame(f'corr_of_{cluster_name}').T
        else:
            empty_row = pd.DataFrame(index=[f'corr_of_{cluster_name}'], columns=results_methods.columns)
        results_methods = pd.concat([results_methods, empty_row])
        rows = [row for row in results_methods.index if 'corr' in row]
        empty_row = results_methods.loc[rows].mean(axis=0).to_frame(f'correlation').T
        results_methods = pd.concat([empty_row, results_methods])
        if not os.path.exists(os.path.join(self.evaluator_path, 'clusters')):
            os.makedirs(os.path.join(self.evaluator_path, 'clusters'))
        if trial is None:
            results_methods.to_csv(
                os.path.join(self.evaluator_path, 'clusters',
                             f'results_methods_{cluster_name}_first.csv'),
                float_format="%.2f")
        else:
            results_methods.to_csv(
                os.path.join(self.evaluator_path, 'clusters',
                             f'results_methods_{cluster_name}_{trial}.csv'),
                float_format="%.2f")


    def evaluate_cluster_func(self, predictions, predictions_eval, y_scaled, y_eval_scaled, cluster_name):
        results_methods = pd.DataFrame()
        results_methods_scaled = pd.DataFrame()
        for method in predictions.keys():
            if method not in predictions.keys():
                continue
            if method not in predictions_eval.keys():
                continue
            pred_train_scaled = predictions[method]
            if pred_train_scaled.shape[0] == 0:
                continue

            pred_eval_scaled = predictions_eval[method]

            pred_train = self.scaler.inverse_transform_data(pred_train_scaled,
                                                            f'target_{self.scale_target_method}')
            if pred_eval_scaled.shape[0] != 0:
                pred_eval = self.scaler.inverse_transform_data(pred_eval_scaled,
                                                               f'target_{self.scale_target_method}')
            else:
                pred_eval = pred_eval_scaled
            path_cluster = self.clusters[cluster_name] if cluster_name != 'global' \
                else os.path.join(self.static_data['path_model'], 'global')
            cv_masks = joblib.load(os.path.join(path_cluster, 'cv_mask.pickle'))
            cv_names = [ 'val', 'test']
            results_methods_temp = pd.DataFrame()
            results_methods_temp_scaled = pd.DataFrame()
            if pred_eval_scaled.shape[0] != 0:
                cv_mask = pred_eval.index.intersection(y_eval_scaled.index)
                res_eval_scaled = compute_metrics(pred_eval_scaled.loc[cv_mask], y_eval_scaled.loc[cv_mask],
                                                  1 if self.rated is not None else None,
                                                  f'{cluster_name}_{method}')['mae'].to_frame()
                y_eval = self.scaler.inverse_transform_data(y_eval_scaled,
                                                            f'target_{self.scale_target_method}')
                res_eval = compute_metrics(pred_eval.loc[cv_mask], y_eval.loc[cv_mask], self.rated,
                                           f'{cluster_name}_{method}')['mae'].to_frame()
                res_eval.columns = [f'{column}_eval' for column in res_eval.columns]
                res_eval_scaled.columns = [f'{column}_eval' for column in res_eval_scaled.columns]
                results_methods_temp = res_eval
                results_methods_temp_scaled = res_eval_scaled

            target_labels = ['target']
            columns = [y_scaled.columns[0]]
            y = self.scaler.inverse_transform_data(y_scaled,
                                                   f'target_{self.scale_target_method}')
            avg_cols = []
            avg_cols_scaled = []
            for col, name_col in zip(columns, target_labels):
                if name_col == 'target':
                    p_scaled = copy.deepcopy(pred_train_scaled)
                    p = copy.deepcopy(pred_train)
                else:
                    raise ValueError(f'{name_col} should implement with resampling enabled')
                cv_mask = np.concatenate(cv_masks[1:])
                cv_mask = pd.DatetimeIndex(cv_mask).intersection(p_scaled.index)
                res_total_scaled = \
                    compute_metrics(p_scaled.loc[cv_mask], y_scaled.loc[cv_mask],
                                    1 if self.rated is not None else None,
                                    f'{cluster_name}_{method}')['mae'].to_frame()
                res_total = compute_metrics(p.loc[cv_mask], y.loc[cv_mask], self.rated,
                                            f'{cluster_name}_{method}')['mae'].to_frame()
                res_total_scaled.columns = [f'{column}_{name_col}_total'
                                            for column in res_total_scaled.columns]
                res_total.columns = [f'{column}_{name_col}_total'
                                     for column in res_total.columns]
                results_methods_temp = pd.concat([results_methods_temp, res_total], axis=1)
                results_methods_temp_scaled = pd.concat([results_methods_temp_scaled, res_total_scaled],
                                                        axis=1)

                for cv_name, cv_mask in zip(cv_names, cv_masks):
                    cv_mask = cv_mask.intersection(p_scaled.index)
                    if cv_mask.shape[0] == 0:
                        continue
                    res_temp_scaled = compute_metrics(p_scaled.loc[cv_mask],
                                                      y_scaled.loc[cv_mask],
                                                      1 if self.rated is not None else None,
                                                      f'{cluster_name}_{method}')['mae'].to_frame()
                    res_temp = compute_metrics(p.loc[cv_mask], y.loc[cv_mask],
                                               self.rated,
                                               f'{cluster_name}_{method}')['mae'].to_frame()
                    res_temp.columns = [f'{column}_{name_col}_{cv_name}'
                                        for column in res_temp.columns]
                    res_temp_scaled.columns = [f'{column}_{name_col}_{cv_name}'
                                               for column in res_temp_scaled.columns]
                    res_temp_scaled.columns = [f'{column}_scaled'
                                               for column in res_temp_scaled.columns]
                    avg_cols.append(res_temp.columns[0])
                    avg_cols_scaled.append(res_temp_scaled.columns[0])
                    results_methods_temp = pd.concat([results_methods_temp, res_temp], axis=1)
                    results_methods_temp_scaled = pd.concat([results_methods_temp_scaled,
                                                             res_temp_scaled],
                                                            axis=1)
            experiment = pd.DataFrame([[cluster_name.split('_')[0], '_'.join(cluster_name.split('_')[1:]),
                                        method.split('_')[0], method.split('_')[-1]]],
                                      index=[f'{cluster_name}_{method}'],
                                      columns=['clusterer', 'cluster', 'method', 'experiment'])
            results_methods_temp['average'] = results_methods_temp[avg_cols].sum(axis=1)
            results_methods_temp_scaled['average'] = results_methods_temp_scaled[avg_cols_scaled].sum(axis=1)
            results_methods_temp = pd.concat([experiment, results_methods_temp], axis=1)
            results_methods_temp_scaled = pd.concat([experiment, results_methods_temp_scaled], axis=1)
            results_methods = pd.concat([results_methods, results_methods_temp])
            results_methods_scaled = pd.concat([results_methods_scaled, results_methods_temp_scaled])
        return results_methods, results_methods_scaled

    def evaluate_methods(self):
        results_methods_all = []
        results_methods_scaled_all = []
        y_scaled = self.feed_target(train=True)
        y_eval_scaled = self.feed_target()
        if 'clusters' in self.predictions.keys():
            for clusterer_method, rules in self.predictions['clusters'].items():
                for cluster_name, methods_predictions in rules.items():
                    results_methods, results_methods_scaled = (
                        self.evaluate_cluster_func(methods_predictions,
                                                   self.predictions_eval['clusters'][clusterer_method][cluster_name],
                                                   y_scaled, y_eval_scaled, cluster_name))
                    results_methods = results_methods.sort_values(by=['method', 'average'])
                    results_methods_scaled = results_methods_scaled.sort_values(by=['method', 'average'])
                    for method in results_methods['method'].unique():
                        file_exp = os.path.join(self.clusters[cluster_name], f'results_{cluster_name}_{method}.csv')
                        df = results_methods.loc[results_methods['method'] == method]
                        df_scaled = results_methods_scaled.loc[results_methods_scaled['method'] == method]
                        experiments = pd.read_csv(file_exp, index_col=0)
                        if 'trial_number' not in experiments.columns:
                            experiments['trial_number'] = experiments.number
                        experiments = experiments.astype({'trial_number': str})
                        df = df.astype({'experiment': str})
                        experiments['trial_number'] = experiments['trial_number'].apply(lambda x: x.split('.')[0])
                        df['experiment'] = df['experiment'].apply(lambda x: x.split('.')[0])
                        df = df.join(experiments.set_index('trial_number'), on='experiment', how='inner')
                        df_scaled = df_scaled.join(experiments.set_index('trial_number'), on='experiment', how='inner')
                        results_methods_all.append(df)
                        results_methods_scaled_all.append(df_scaled)
                    empty_row = pd.DataFrame(index=[f'corr_of_{cluster_name}'], columns=results_methods.columns)
                    empty_row_scaled = pd.DataFrame(index=[f'corr_of_{cluster_name}'],
                                                    columns=results_methods_scaled.columns)
                    results_methods_all.append(empty_row)
                    results_methods_scaled_all.append(empty_row_scaled)
        if len(self.global_methods) > 0:
            cluster_name = 'global'
            results_methods, results_methods_scaled = (
                self.evaluate_cluster_func(self.predictions['global'],
                                           self.predictions_eval['global'], y_scaled, y_eval_scaled,
                                           'global'))
            results_methods = results_methods.sort_values(by=['method', 'average'])
            results_methods_scaled = results_methods_scaled.sort_values(by=['method', 'average'])
            for method in results_methods['method'].unique():
                path_global = os.path.join(self.static_data['path_model'], 'global')
                file_exp = os.path.join(path_global, f'results_{cluster_name}_{method}.csv')
                df = results_methods.loc[results_methods['method'] == method]
                df_scaled = results_methods_scaled.loc[results_methods_scaled['method'] == method]
                experiments = pd.read_csv(file_exp, index_col=0)
                if 'trial_number' not in experiments.columns:
                    experiments['trial_number'] = experiments.number
                experiments = experiments.astype({'trial_number': str})
                df = df.astype({'experiment': str})
                experiments['trial_number'] = experiments['trial_number'].apply(lambda x: x.split('.')[0])
                df['experiment'] = df['experiment'].apply(lambda x: x.split('.')[0])
                df = df.join(experiments.set_index('trial_number'), on='experiment', how='inner')
                df_scaled = df_scaled.join(experiments.set_index('trial_number'), on='experiment', how='inner')
                results_methods_all.append(df)
                results_methods_scaled_all.append(df_scaled)
            empty_row = pd.DataFrame(index=[f'corr_of_{cluster_name}'], columns=results_methods.columns)
            empty_row_scaled = pd.DataFrame(index=[f'corr_of_{cluster_name}'],
                                            columns=results_methods_scaled.columns)
            results_methods_all.append(empty_row)
            results_methods_scaled_all.append(empty_row_scaled)
        results_methods_all = pd.concat(results_methods_all)
        results_methods_scaled_all = pd.concat(results_methods_scaled_all)
        results_methods_all.to_csv( os.path.join(self.evaluator_path, f'results_methods.csv'), float_format="%.3f")

        results_methods_scaled_all.to_csv(os.path.join(self.evaluator_path, f'results_methods_scaled.csv'),
                                               float_format="%.3f")

    def evaluate_clusterer(self, pred_dict, y, y_scaled):
        eval_metrics = pd.DataFrame()
        eval_metrics_scaled = pd.DataFrame()
        for clusterer_name, clusterer_pred_scaled in pred_dict.items():
            clusterer_pred_scaled = clusterer_pred_scaled.mean(axis=1).to_frame(f'{clusterer_name}_clusterer')
            clusterer_pred_scaled, y = sync_datasets(clusterer_pred_scaled, y)
            y_scaled = y_scaled.loc[y.index]
            clusterer_pred = self.scaler.inverse_transform_data(clusterer_pred_scaled,
                                                                f'target_{self.scale_target_method}')
            eval_metrics = pd.concat([eval_metrics, compute_metrics(clusterer_pred, y, self.rated,
                                                                    f'{clusterer_name}_clusterer')])
            eval_metrics_scaled = pd.concat([eval_metrics_scaled, compute_metrics(clusterer_pred_scaled, y_scaled,
                                                                                  1 if self.rated is not None else None,
                                                                                  f'{clusterer_name}_clusterer')])
            empty_row = pd.DataFrame(columns=eval_metrics.columns, index=[f'{clusterer_name}_clusterer_ends'])
            eval_metrics = pd.concat([eval_metrics, empty_row])
            eval_metrics_scaled = pd.concat([eval_metrics_scaled, empty_row])
        return eval_metrics, eval_metrics_scaled

    def evaluate_cluster_averages(self, pred_dict, y, y_scaled):
        eval_metrics = pd.DataFrame()
        eval_metrics_scaled = pd.DataFrame()
        for clusterer_name, cluster_group_pred in pred_dict.items():
            if 'averages' in cluster_group_pred.keys():
                for method, method_pred_scaled in cluster_group_pred['averages'].items():
                    method_pred_scaled, y_slice = sync_datasets(method_pred_scaled, y)
                    y_scaled_slice = y_scaled.loc[y_slice.index]
                    method_pred = self.scaler.inverse_transform_data(method_pred_scaled,
                                                                     f'target_{self.scale_target_method}')
                    eval_metrics = pd.concat([eval_metrics, compute_metrics(method_pred, y_slice, self.rated,
                                                                            f'{clusterer_name}_{method}',
                                                                            multi_output=self.multi_output)])
                    eval_metrics_scaled = pd.concat([eval_metrics_scaled, compute_metrics(method_pred_scaled, y_scaled_slice,
                                                                                          1 if self.rated is not None else None,
                                                                                          f'{clusterer_name}_{method}',
                                                                                          multi_output=self.multi_output)])
                empty_row = pd.DataFrame(columns=eval_metrics.columns, index=[f'{clusterer_name}_ends'])
                eval_metrics = pd.concat([eval_metrics, empty_row])
                eval_metrics_scaled = pd.concat([eval_metrics_scaled, empty_row])
            for cluster_name, cluster_pred_scaled in cluster_group_pred.items():
                for method, method_pred_scaled in cluster_pred_scaled.items():
                    method_pred_scaled, y_slice = sync_datasets(method_pred_scaled, y)
                    y_scaled_slice = y_scaled.loc[y_slice.index]
                    method_pred = self.scaler.inverse_transform_data(method_pred_scaled,
                                                                     f'target_{self.scale_target_method}')
                    eval_metrics = pd.concat([eval_metrics, compute_metrics(method_pred, y_slice, self.rated,
                                                                            f'{clusterer_name}_{cluster_name}_{method}',
                                                                            multi_output=self.multi_output)])
                    eval_metrics_scaled = pd.concat([eval_metrics_scaled, compute_metrics(method_pred_scaled, y_scaled_slice,
                                                                                          1 if self.rated is not None else None,
                                                                                          f'{clusterer_name}_{cluster_name}_{method}',
                                                                                          multi_output=self.multi_output)])
                empty_row = pd.DataFrame(columns=eval_metrics.columns, index=[f'{clusterer_name}_{cluster_name}_ends'])
                eval_metrics = pd.concat([eval_metrics, empty_row])
                eval_metrics_scaled = pd.concat([eval_metrics_scaled, empty_row])
        return eval_metrics, eval_metrics_scaled

    def evaluate_distributed(self, pred_dict, y, y_scaled):
        eval_metrics = pd.DataFrame()
        eval_metrics_scaled = pd.DataFrame()
        for distributed_name, distributed_pred_scaled in pred_dict.items():
            distributed_pred_scaled, y = sync_datasets(distributed_pred_scaled, y)
            y_scaled = y_scaled.loc[y.index]
            distributed_pred = self.scaler.inverse_transform_data(distributed_pred_scaled,
                                                                  f'target_{self.scale_target_method}')
            eval_metrics = pd.concat([eval_metrics, compute_metrics(distributed_pred, y, self.rated,
                                                                    f'{distributed_name}_model',
                                                                    multi_output=self.multi_output)])
            eval_metrics_scaled = pd.concat([eval_metrics_scaled, compute_metrics(distributed_pred_scaled, y_scaled,
                                                                                  1 if self.rated is not None else None,
                                                                                  f'{distributed_name}_model',
                                                                                  multi_output=self.multi_output)])
            empty_row = pd.DataFrame(columns=eval_metrics.columns, index=[f'{distributed_name}_model_ends'])
            eval_metrics = pd.concat([eval_metrics, empty_row])
            eval_metrics_scaled = pd.concat([eval_metrics_scaled, empty_row])
        return eval_metrics, eval_metrics_scaled

    def evaluate_combining_models(self, pred_dict, y, y_scaled):
        eval_metrics = pd.DataFrame()
        eval_metrics_scaled = pd.DataFrame()
        for combining_model_name, combining_model_pred_scaled in pred_dict.items():
            combining_model_pred_scaled, y = sync_datasets(combining_model_pred_scaled, y)
            y_scaled = y_scaled.loc[y.index]
            combining_model_pred = self.scaler.inverse_transform_data(combining_model_pred_scaled,
                                                                      f'target_{self.scale_target_method}')
            eval_metrics = pd.concat([eval_metrics, compute_metrics(combining_model_pred, y, self.rated,
                                                                    f'{combining_model_name}_model',
                                                                    multi_output=self.multi_output)])
            eval_metrics_scaled = pd.concat([eval_metrics_scaled, compute_metrics(combining_model_pred_scaled, y_scaled,
                                                                                  1 if self.rated is not None else None,
                                                                                  f'{combining_model_name}_model',
                                                                                  multi_output=self.multi_output)])
            empty_row = pd.DataFrame(columns=eval_metrics.columns, index=[f'{combining_model_name}_model_ends'])
            eval_metrics = pd.concat([eval_metrics, empty_row])
            eval_metrics_scaled = pd.concat([eval_metrics_scaled, empty_row])
        return eval_metrics, eval_metrics_scaled

    def evaluate_models(self, special_index=False):
        if (not os.path.exists(os.path.join(self.evaluator_path, 'results_methods.csv')) and
            not os.path.exists(os.path.join(self.evaluator_path, 'results_methods_scaled.csv'))) or \
                self.refit:
            results = pd.DataFrame()
            results_scaled = pd.DataFrame()
            y_scaled = self.feed_target(train=True, inverse=False)
            y = self.scaler.inverse_transform_data(y_scaled,
                                                   f'target_{self.scale_target_method}')
            y_eval_scaled = self.feed_target()
            y_eval = self.scaler.inverse_transform_data(y_eval_scaled,
                                                        f'target_{self.scale_target_method}')
            for model_name, model_preds in self.predictions.items():
                if model_name == 'clusterer':
                    res, res_scaled = self.evaluate_clusterer(model_preds, y, y_scaled)
                elif model_name == 'clusters':
                    res, res_scaled = self.evaluate_cluster_averages(model_preds, y, y_scaled)
                elif model_name == 'global':
                    res, res_scaled = self.evaluate_distributed(model_preds, y, y_scaled)
                elif model_name == 'models':
                    res, res_scaled = self.evaluate_combining_models(model_preds, y, y_scaled)
                else:
                    raise ValueError(f'Unknown model for evaluation {model_name}')
                results = pd.concat([results, res])
                results_scaled = pd.concat([results_scaled, res_scaled])
                results.to_csv(os.path.join(self.evaluator_path, f'results_models_train.csv'), float_format="%.2f")
                results_scaled.to_csv(os.path.join(self.evaluator_path, f'results_models_train_scaled.csv'),
                                      float_format="%.2f")
                results_eval = pd.DataFrame()
                results_eval_scaled = pd.DataFrame()
                if special_index:
                    file_manager = FilesManager(self.static_data, is_online=False, train=False)
                    data_row = file_manager.check_if_exists_row_data()
                    dates_special = data_row[data_row['sp_index'] > 10].index
                    dates_special = dates_special.intersection(y_eval.index)
                    y_eval = y_eval.loc[dates_special]
                    y_eval_scaled = y_eval_scaled.loc[dates_special]
                for model_name, model_preds in self.predictions_eval.items():
                    if model_name == 'clusterer':
                        res, res_scaled = self.evaluate_clusterer(model_preds, y_eval, y_eval_scaled)
                    elif model_name == 'clusters':
                        res, res_scaled = self.evaluate_cluster_averages(model_preds, y_eval, y_eval_scaled)
                    elif model_name == 'global':
                        res, res_scaled = self.evaluate_distributed(model_preds, y_eval, y_eval_scaled)
                    elif model_name == 'models':
                        res, res_scaled = self.evaluate_combining_models(model_preds, y_eval, y_eval_scaled)
                    else:
                        raise ValueError(f'Unknown model for evaluation {model_name}')

                    results_eval = pd.concat([results_eval, res])
                    results_eval_scaled = pd.concat([results_eval_scaled, res_scaled])

                    results_models_eval_file = f'results_models_eval.csv'
                    results_eval_scaled_file = f'results_models_eval_scaled.csv'
                    results_eval.to_csv(os.path.join(self.evaluator_path, results_models_eval_file),
                                        float_format="%.6f")
                    results_eval_scaled.to_csv(os.path.join(self.evaluator_path, results_eval_scaled_file),
                                               float_format="%.6f")

    def evaluate_averages(self):
        if 'clusters' not in self.predictions_eval.keys():
            return
        results = pd.DataFrame()
        results_scaled = pd.DataFrame()
        y_scaled = self.feed_target(train=True)
        y = self.scaler.inverse_transform_data(y_scaled,
                                               f'target_{self.scale_target_method}')
        y_eval_scaled = self.feed_target()
        y_eval = self.scaler.inverse_transform_data(y_eval_scaled,
                                                    f'target_{self.scale_target_method}')
        res, res_scaled = self.evaluate_cluster_averages(self.predictions_eval['clusters'], y_eval, y_eval_scaled)
        res.columns = [f'{col}_eval' for col in res.columns]
        res_scaled.columns = [f'{col}_eval' for col in res_scaled.columns]
        results = pd.concat([results, res])
        results_scaled = pd.concat([results_scaled, res_scaled])
        results.to_csv(os.path.join(self.evaluator_path, f'results_averages_eval.csv'),
                            float_format="%.6f")
        results_scaled.to_csv(os.path.join(self.evaluator_path, f'results_averages_eval_scaled.csv'),
                                   float_format="%.6f")

        res, res_scaled = self.evaluate_cluster_averages(self.predictions['clusters'], y, y_scaled)
        results = pd.concat([results, res])
        results_scaled = pd.concat([results_scaled, res_scaled])
        results.to_csv(os.path.join(self.evaluator_path, f'results_averages_train.csv'),
                       float_format="%.6f")
        results_scaled.to_csv(os.path.join(self.evaluator_path, f'results_averages_train_scaled.csv'),
                              float_format="%.6f")