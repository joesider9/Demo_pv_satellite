import copy
import os
import joblib
import torch
import cv2

import numpy as np
import pandas as pd
from einops import rearrange
from einops import repeat

from eforecast.common_utils.nwp_utils import get_clear_sky
from eforecast.common_utils.nwp_utils import bcs

from eforecast.datasets.files_manager import FilesManager


class DataTransformer:
    def __init__(self, static_data, recreate=False, is_online=False, train=False):
        self.transformers = dict()
        self.online = is_online
        self.train = train
        self.filename = os.path.join(static_data['path_model'], 'transformers.pickle')
        if os.path.exists(self.filename):
            try:
                self.transformers = joblib.load(self.filename)
            except:
                self.transformers = dict()
                os.remove(self.filename)
        if recreate:
            self.transformers = dict()
            if os.path.exists(self.filename):
                os.remove(self.filename)
        self.static_data = static_data
        self.variables_index = {var_data['name']: var_data['transformer']
                                for var_data in self.static_data['variables']
                                if var_data['transformer'] is not None}
        self.transformer_params = {var_data['name']: var_data['transformer_params']
                                   for var_data in self.static_data['variables']
                                   if var_data['transformer_params'] is not None}
        self.coord = self.static_data['coord']
        self.local_timezone = self.static_data['local_timezone']
        self.site_timezone = self.static_data['site_timezone']
        self.ts_resolution = self.static_data['ts_resolution']
        self.nwp_data_merge = self.static_data['nwp_data_merge']
        self.nwp_data_compress = self.static_data['compress_data']
        self.files_manager = FilesManager(static_data, is_online=is_online)

    def save(self):
        joblib.dump(self.transformers, self.filename)

    def fit(self, data, variable, data_dates=None):
        if variable not in self.transformers.keys():
            self.transformers[variable] = dict()
        transformations = self.variables_index[variable]
        if not isinstance(transformations, list):
            transformations = [transformations]
        for transformation in transformations:
            if transformation not in {'inverse', 'brightContrast', 'sum', 'eq_histogram', 'grey', 'sam', 'normalize'}:
                if transformation == 'clear_sky':
                    if isinstance(data, pd.Series):
                        data = data.to_frame()
                    if isinstance(data, pd.DataFrame):
                        dates = data.index
                    else:
                        if data_dates is None:
                            raise ValueError('If data is not dataframe, data_dates should be provided')
                        dates = data_dates
                    ghi = get_clear_sky(dates, self.coord[0], self.coord[1], self.local_timezone, self.site_timezone,
                                        self.ts_resolution)
                    self.transformers[variable][transformation] = {'max': ghi.max(),
                                                         'values': ghi}
                elif transformation == 'resize':
                    self.transformers[variable][transformation] = self.transformer_params[variable]['resize']
                elif transformation == 'crop':
                    params = self.transformer_params[variable]
                    if params is None:
                        raise ValueError('If transformation is crop, params crop or resize should be provided')
                    crop = params['crop']
                    self.transformers[variable][transformation] = {
                                                            'lat_min': crop[0][0],
                                                            'lat_max': crop[0][1],
                                                            'long_min': crop[1][0],
                                                            'long_max': crop[1][1],
                                                         }
                elif transformation == 'fillnan':
                    params = self.transformer_params[variable]
                    if params is None:
                        raise ValueError('If transformation is crop, params crop or resize should be provided')
                    self.transformers[variable][transformation] = {'value': params['fillnan']}
                elif transformation == 'alignment':
                    params = self.transformer_params[variable]
                    self.transformers[variable][transformation] = params['alignment']
                elif transformation == 'norm_minmax':
                    params = self.transformer_params[variable]
                    if transformation not in self.transformers[variable].keys():
                        self.transformers[variable][transformation] = dict()
                    self.transformers[variable][transformation] = params['norm_minmax']
                else:
                    raise NotImplementedError(f'{transformation} transformation is not implemented yet')
        self.save()

    # def init_segment_anything(self, variable):
    #     if not hasattr(self, 'sam'):
    #         sam_checkpoint = self.transformer_params[variable]['sam']
    #         model_type = "vit_h"
    #
    #         self.sam = []
    #         self.mask_generator = []
    #         for gpu in range(self.static_data['n_gpus']):
    #             device = f"cuda:{gpu}"
    #             self.sam.append(sam_model_registry[model_type](checkpoint=sam_checkpoint))
    #             self.sam[gpu].to(device=device)
    #             self.mask_generator.append(SamAutomaticMaskGenerator(self.sam[gpu]))
    #         self.mask_annotator = sv.MaskAnnotator(color_lookup=sv.ColorLookup.INDEX)

    def transform(self, data, variable, data_dates=None, gpu=0):
        if variable not in self.variables_index.keys():
            return data
        transformations = self.variables_index[variable]
        if not isinstance(transformations, list):
            transformations = [transformations]
        for transformation in transformations:
            if transformation not in self.transformers[variable].keys() and transformation not in {'inverse',
                                          'brightContrast', 'sum', 'eq_histogram', 'grey', 'sam', 'optical_flow',
                                                                                                   'normalize'}:
                self.update(data, variable, data_dates=data_dates)
            if transformation == 'clear_sky':
                if isinstance(data, pd.Series):
                    data = data.to_frame()
                if isinstance(data, pd.DataFrame):
                    dates = data.index
                else:
                    if data_dates is None:
                        raise ValueError('If data is not dataframe, data_dates should be provided')
                    dates = data_dates
                ghi = self.transformers[variable][transformation]['values']
                dates_diff = dates.difference(ghi.index)
                if dates_diff.shape[0] > 0:
                    ghi_new = get_clear_sky(dates_diff, self.coord[0], self.coord[1], self.local_timezone,
                                            self.site_timezone, self.ts_resolution)
                    ghi = pd.concat([ghi, ghi_new])
                    ghi = ghi.sort_index()
                ghi = ghi[~ghi.index.duplicated()]
                ghi = ghi.loc[dates]

                rate = np.tile(np.expand_dims((self.transformers[variable][transformation]['max'] / ghi).values,
                                              axis=[i for i in range(data.ndim - 1, 1, -1)]),
                               [1] + list(data.shape[1:]))
                data = rate * data
                data[data < 0] = 0
                data = data.astype(np.float32)
            elif transformation == 'crop':
                lat_min = self.transformers[variable][transformation]['lat_min']
                lat_max = self.transformers[variable][transformation]['lat_max']
                long_min = self.transformers[variable][transformation]['long_min']
                long_max = self.transformers[variable][transformation]['long_max']
                if lat_max > data.shape[1] or long_max > data.shape[2]:
                    print('Cannot crop images')
                else:
                    data = data[:, lat_min:lat_max, :][:, :, long_min:long_max]
                data = data.astype(np.float32)
            elif transformation == 'inverse':
                data = data[:, ::-1, :]
                data = data.astype(np.float32)
            elif transformation == 'resize':
                resize = self.transformers[variable][transformation][::-1]
                if data.shape[2] != resize[0] or data.shape[1] != resize[1]:
                    data1 = copy.deepcopy(data)
                    data_transformed = []
                    for n in range(data1.shape[0]):
                        img_interp = cv2.resize(data1[n], resize)
                        data_transformed.append(img_interp)
                    data = np.array(data_transformed)
                    data = data.astype(np.float32)
            elif transformation == 'fillnan':
                data[data <= 0] = self.transformers[variable][transformation]['value']
                data = np.nan_to_num(data, nan=self.transformers[variable][transformation]['value'])
                data1 = copy.deepcopy(data)
                data_transformed = []
                for n in range(data1.shape[0]):
                    img_norm =data1[n]
                    q = np.quantile(img_norm, 0.9999)
                    img_norm[img_norm > q] = q
                    data_transformed.append(img_norm)
                data = np.array(data_transformed)
                data = data.astype(np.float32)
            elif transformation == 'alignment':
                coords = np.array(self.transformers[variable][transformation], dtype = "float32")
                data = rearrange(data.astype('float'), 'c w b -> w b c')
                maxHeight = data.shape[1]
                maxWidth = data.shape[0]
                coords_out = np.float32([[0, 0],
                                         [0, maxHeight],
                                         [maxWidth, maxHeight],
                                         [maxWidth - 1, 0]])
                M = cv2.getPerspectiveTransform(coords, coords_out)
                data_transformed = cv2.warpPerspective(data.astype('float'), M, (maxWidth, maxHeight))
                data_transformed = rearrange(data_transformed, 'w b c -> c w b')
                data = data_transformed.astype(np.float32)
            elif transformation == 'normalize':
                data1 = copy.deepcopy(data)
                data1 = np.clip(data1, 0, None)
                data_transformed = (255 * (data1 / (data1.max() + 0.001)))
                data = np.array(data_transformed)
                data = data.astype(np.float32)
            elif transformation == 'norm_minmax':
                minmax = self.transformers[variable][transformation]
                data1 = copy.deepcopy(data)
                data_transformed = []
                for n in range(data1.shape[0]):
                    if np.max(data1[n]) > minmax[n][1]:
                        print(f'Maximum should set {np.max(data1[n])} for {n} of {variable}')
                        return None
                    if np.min(data1[n]) < minmax[n][0]:
                        print(f'Minimum should set {np.min(data1[n])} for {n} of {variable}')
                        return None
                    img_norm = (255 * (data1[n] - minmax[n][0]) / (minmax[n][1] - minmax[n][0])).astype('uint8')
                    data_transformed.append(img_norm)
                data = np.array(data_transformed)
                data = data.astype(np.float32)
            elif transformation == 'brightContrast':
                data1 = copy.deepcopy(data)
                data_transformed = []
                for n in range(data1.shape[0]):
                    img_norm = bcs(data1[n])
                    if img_norm is None:
                        return None
                    data_transformed.append(img_norm)
                data = np.array(data_transformed)
                data = data.astype(np.float32)
            elif transformation == 'eq_histogram':
                data1 = copy.deepcopy(data)
                data_transformed = []
                for n in range(data1.shape[0]):
                    img_norm = cv2.equalizeHist(data1[n].astype(np.uint8))
                    if img_norm is None:
                        return None
                    data_transformed.append(img_norm)
                data = np.array(data_transformed)
                data = data.astype(np.float32)
            elif transformation == 'sum':
                # data = np.expand_dims(np.sum(data, axis=0), axis=0)
                data1 = copy.deepcopy(data)
                data_transformed = np.zeros(data1.shape[1:])
                max_sum = 0
                for n in range(data1.shape[0]):
                    data_transformed += data1[n] / np.maximum(0.01, np.max(data1[n]))
                    max_sum += np.maximum(0.01, np.max(data1[n]))
                data = max_sum * np.expand_dims(data_transformed, axis=0) / 6
                data = data.astype(np.float32)
            # elif transformation == 'sam':
            #     data1 = data.copy()
            #     shape = data1.shape
            #     if shape[0] == 3:
            #         data1 = rearrange(data1, 'c w h -> w h c')
            #     elif shape[-1] == 3:
            #         pass
            #     else:
            #         raise ValueError('The image should have 3 channels RGB')
            #     masks = self.mask_generator[gpu].generate(data1)
            #     detections = sv.Detections.from_sam(masks)
            #     data1 = self.mask_annotator.annotate(data1, detections)
            #     if shape[0] == 3:
            #         data1 = rearrange(data1, 'w h c -> c w h')
            #     data = data1.copy()
            #     data = data.astype(np.float32)
            else:
                raise NotImplementedError(f'{transformation} transformation is not implemented yet')
        return data

    def update(self, data, variable, data_dates=None):
        print(f'Update imputer')
        if not self.online:
            self.fit(data, variable, data_dates=data_dates)
