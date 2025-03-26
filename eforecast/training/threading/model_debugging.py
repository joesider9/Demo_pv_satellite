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



path_weights = ('D:/models/FA_ellados/FA_group_ver1/FA/FA_morning/multi-output/model_ver1/global/LSTM/test_1')
gpu_id = 'cpu'
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
        cv_masks = [cv_masks[i] for i in [0, 2, 1]]
        self.model = DeepNetwork(self.static_data, path_weights, self.params, train=True, refit=self.refit)
        try:
            self.model.fit(cv_masks, gpu_id=gpu_id)
        except Exception as e:
            tb = traceback.format_exception(e)
            print("".join(tb))
            with open(os.path.join(path_weights, 'error.txt'), mode='w') as fp:
                fp.write(" ".join(tb))
            raise Exception("".join(tb))


if __name__ == '__main__':
    # path_cluster = os.path.dirname(os.path.dirname(path_weights))
    # static_data = joblib.load(os.path.join(path_cluster, 'static_data.pickle'))
    from morning.configuration.config import config
    from eforecast.init.initialize import initializer
    static_data = initializer(config())
    params = joblib.load(os.path.join(path_weights, 'parameters.pickle'))
    trainer = Trainer(static_data, params, refit)
    trainer.train(gpu_id)

