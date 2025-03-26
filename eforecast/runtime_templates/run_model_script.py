import importlib
from eforecast.init.initialize import initializer
from eforecast.datasets.dataset_creator import DatasetCreator
from eforecast.clustering.clustering_manager import ClusterOrganizer

from eforecast.prediction.predict import Predictor

def create_datasets(static_data, dates):
    dataset = DatasetCreator(static_data, train=False, is_online=True, dates=dates)
    if static_data['NWP'] is not None:
        dataset.create_nwp_dataset(parallel=False)
    if static_data['use_image']:
        dataset.create_image_dataset(parallel=False)
    dataset.create_row_datasets()
    dataset.create_lstm_dataset()


def predict(dates, project_name_list, best_method_list):
    name = '.'.join(project_name_list)
    module = importlib.import_module(f'{name}.configuration.config')
    config = getattr(module, 'config')

    static_data = initializer(config(), online=True)
    cluster_organizer = ClusterOrganizer(static_data, is_online=True, train=False, refit=False)
    cluster_organizer.update_cluster_folders()
    create_datasets(static_data, dates)
    predictor = Predictor(static_data, train=False, online=True)
    predictor.predict_regressors(average=True, parallel=True)
    predictor.predict_combine_methods()
    predictor.compute_predictions_averages()
    predictor.predict_combine_models(combine_methods=['kmeans'])
    if len(best_method_list) == 2:
        return predictor.inverse_transform_predictions(predictor.predictions[best_method_list[0]][best_method_list[1]])
    elif len(best_method_list) == 4:
        return predictor.inverse_transform_predictions(predictor.predictions[best_method_list[0]][best_method_list[1]]
                                                       [best_method_list[2]][best_method_list[3]])