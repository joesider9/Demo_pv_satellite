import copy
import os
import sys

path_pycharm = (os.path.normpath(os.path.dirname(__file__))).split(os.sep)
path_pycharm = os.path.join(*path_pycharm[:-3])
if sys.platform == 'linux':
    path_pycharm = '/' + path_pycharm
print(path_pycharm)
sys.path.append(path_pycharm)

import joblib
import traceback
from eforecast.common_utils.train_utils import send_predictions

try:
    import torch
    from eforecast.deep_models.pytorch_2x.network import DeepNetwork
    print('Backend is set pytorch')
except:
    try:
        from eforecast.deep_models.tf_2x.network import DeepNetwork
        from eforecast.deep_models.tf_2x.transformers.tranformer_network import TransformerNetwork
        print('Backend is set Tensorflow 2.10')
    except:
        raise ImportError('Cannot find backend')

# path_weights = sys.argv[1]
# gpu_id = int(sys.argv[2])
# refit = bool(sys.argv[3])


path_weights = ('/media/smartrue/HHD1/George/models/PPC/PPC_sat_ver2/pv/site_/multi-output/model_ver0/'
                'global/CNN/test_1')
gpu_id = 1
refit = bool(1)


class Trainer:
    def __init__(self, static_data, params, refit):
        self.params = params
        self.static_data = static_data
        self.method = self.params['method']
        self.refit = refit
        self.cluster_name = self.params['name']
        self.cluster_dir = self.params['cluster_dir']
        self.use_image = self.static_data['use_image']


    def train(self, gpu_id):
        cv_masks = joblib.load(os.path.join(self.cluster_dir, 'cv_mask.pickle'))
        self.model = DeepNetwork(self.static_data, path_weights, self.params, train=True, refit=self.refit)
        self.model.fit(cv_masks, gpu_id=gpu_id)

    def evaluate(self):
        from eforecast.datasets.data_feeder import DataFeeder
        from eforecast.common_utils.eval_utils import compute_metrics
        self.model = DeepNetwork(self.static_data, path_weights, train=False)
        pred = self.model.predict()
        data_feeder = DataFeeder(self.static_data, train=False)
        y = data_feeder.feed_target(inverse=False)
        dates = pred.index.intersection(y.index)
        res_eval = compute_metrics(pred.loc[dates], y.loc[dates], 1,
                                   f'global_CNN')['mae'].to_frame()
        print(res_eval)


if __name__ == '__main__':
    path_cluster = os.path.dirname(os.path.dirname(path_weights))
    # static_data = joblib.load(os.path.join(path_cluster, 'static_data.pickle'))
    from site_.short_term_image.configuration.config import config

    from eforecast.init.initialize import initializer
    static_data = initializer(config())
    params = joblib.load(os.path.join(path_weights, 'parameters.pickle'))
    trainer = Trainer(static_data, params, refit)
    trainer.train(gpu_id)
    trainer.evaluate()

