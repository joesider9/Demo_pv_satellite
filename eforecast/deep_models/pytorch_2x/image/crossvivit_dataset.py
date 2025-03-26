import os
from tqdm import tqdm
import random
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
import cv2
import numpy as np
import pandas as pd
import joblib
import torch
from torchvision.models.optical_flow import Raft_Large_Weights
from torchvision.models.optical_flow import raft_large
from torchvision.utils import flow_to_image
import torchvision.transforms.functional as F
from einops import rearrange
from einops import repeat

class ImageDataloader:
    def __init__(self, dataset, batch_size, num_workers):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.length = len(self.dataset.cv_mask)
        self.n_batches = int(self.length / self.batch_size) + 1
        self.dates = set()
        self.valid = True

    def reset(self):
        self.dates = set()
        self.valid = True

    def get_batch(self, device=torch.device('cpu'), randomly=True):
        random.seed(random.randint(1, 100))
        dates = set(self.dataset.cv_mask).difference(self.dates)
        if len(dates) == 0:
            self.valid = False
        if len(dates) > self.batch_size:
            if randomly:
                dates = random.sample(dates, k=self.batch_size)
            else:
                dates = set(list(dates)[:self.batch_size ])
        self.dates = self.dates.union(dates)

        # batch = [self.dataset.get_safe(date) for date in tqdm(dates)]
        batch = []
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = [executor.submit(self.dataset.get_safe, date) for date in tqdm(dates)]
        for future in as_completed(futures):
            batch.append(future.result())
        batch = [b for b in batch if b is not None]

        if len(batch) == 0:
            self.valid = False
        batch_dict = dict()
        dates_list = []
        for sample in batch:
            if sample is not None:
                for key in sample.keys():
                    if key not in batch_dict.keys() and key != 'date':
                        batch_dict[key] = []
                    if key == 'date':
                        dates_list.append(sample[key])
                    else:
                        batch_dict[key].append(sample[key].unsqueeze(0))

        for key in batch_dict.keys():
            batch_dict[key] = torch.cat(batch_dict[key], 0).to(device)
        batch_dict['dates'] = pd.DatetimeIndex(dates_list)
        return batch_dict



class ImageDataset:
    def __init__(self, static_data, cv_mask, params, use_target=True):
        self.use_target = use_target
        self.path_sat = static_data['sat_folder']
        self.horizon = static_data['horizon']
        self.lat, self.long = static_data['site_indices']
        self.area_adjust = params['area_adjust']
        self.lat, self.long = params['site_indices']
        self.image_type = params['image_type'].split(':')
        self.type = static_data['type']
        self.lags = [var_data for var_data in static_data['variables'] if var_data['type'] == 'image'][0]['lags']
        self.ts_resolution = static_data['ts_resolution']
        self.cv_mask = cv_mask
        self.init_llms()
        self.spatial_coords = self.get_spatial_coords(static_data)

    def get_spatial_coords(self, params):
        site_coord = params['coord']
        site_coord = np.expand_dims(np.array(site_coord), (1, 2))
        spatial_coord = params['image_coord']
        image_size = params['image_size']
        lat = np.linspace(spatial_coord[0], spatial_coord[1], image_size[0])
        lon = np.linspace(spatial_coord[2], spatial_coord[3], image_size[1])
        spatial_coords = np.stack(np.meshgrid(lon, lat)[::-1], axis=0)
        site_coord = repeat(site_coord, 'n w h -> n (w k) (h m)', k=spatial_coords.shape[1],
                            m=spatial_coords.shape[2])

        data = spatial_coords - site_coord
        data = np.power(data[0], 2) + np.power(data[1], 2)
        data = data[(None,) * 3 + (...,)]
        data = repeat(data, 'b t c w h -> b (t k) c w h', k=len(self.lags))
        return data.astype(np.float32)

    def __len__(self) -> int:
        return self.cv_mask.shape[0]

    def get_safe(self, date):
        try:
            return self.get_image(date)
        except:
            return None


    def init_llms(self):
        weights = Raft_Large_Weights.DEFAULT
        self.raft_transforms = weights.transforms()
        # If you can, run this example on a GPU, it will be a lot faster.
        device = "cpu"
        self.raft_model = raft_large(weights=Raft_Large_Weights.DEFAULT, progress=False).to(device)
        self.raft_model = self.raft_model.eval()

        self.sr = cv2.dnn_superres.DnnSuperResImpl_create()
        path = os.path.join(self.path_sat, "FSRCNN_x4.pb")  # "FSRCNN_x4.pb"
        self.sr.readModel(path)
        self.sr.setModel("fsrcnn", 4)

    def upscaled_image(self, sr, image):
        return np.concatenate(
            [np.expand_dims(sr.upsample(image[:, :, i]), axis=-1) for i in range(image.shape[-1])], -1)

    def raft(self, img1_batch, img2_batch):
        device = "cpu"
        shape = img1_batch.shape
        img1_batch = F.resize(img1_batch, size=[256, 256], antialias=False)
        img2_batch = F.resize(img2_batch, size=[256, 256], antialias=False)
        img1_batch, img2_batch = self.raft_transforms(img1_batch, img2_batch)
        predicted_flows = self.raft_model(img1_batch.to(device), img2_batch.to(device))[-1]
        flow_imgs = flow_to_image(predicted_flows)
        flow_imgs = F.resize(flow_imgs[:, 1:, :, :], size=[shape[-2], shape[-1]], antialias=False).detach().numpy()
        return flow_imgs


    def get_optical_flow(self, images):
        img1_batch = torch.from_numpy(images[:-1])
        img2_batch = torch.from_numpy(images[1:])
        data = self.raft(img1_batch, img2_batch)
        return data.astype(np.float32)

    def get_image_grey(self, images):
        inp_lag = []
        for j in range(images.shape[0]):
            sat = images[j, :, :, :]
            sat = np.expand_dims(cv2.cvtColor(sat.astype(np.float32), cv2.COLOR_BGR2GRAY), axis=-1)
            inp_lag.append(np.expand_dims(sat, axis=0))
        return np.concatenate(inp_lag, axis=0)

    def super_resolution(self, image):
        image_res = []

        for i in range(image.shape[-1]):
            img = image[i]
            shape = img.shape
            result = self.upscaled_image(self.sr, img.astype('uint8'))
            lat1 = int(4 * (self.lat - 2 * self.area_adjust))
            long1 = int(4 * (self.long - 2 * self.area_adjust))
            lat2 = int(4 * (self.lat + 2 * self.area_adjust))
            long2 = int(4 * (self.long + 2 * self.area_adjust))
            result = result[lat1:lat2, :, :][:, long1:long2, :]
            result = np.concatenate(
                [np.expand_dims(cv2.resize(result[:, :, i],
                                           dsize=shape[:-1][::-1],
                                           interpolation=cv2.INTER_CUBIC), axis=-1)
                for i in range(result.shape[-1])], -1)
            if self.type == 'image2image':
                img_crop = result
            else:
                result = self.upscaled_image(self.sr, result)
                lat1 = int(4 * (self.lat - 2 * self.area_adjust))
                long1 = int(4 * (self.long - 2 * self.area_adjust))
                lat2 = int(4 * (self.lat + 2 * self.area_adjust))
                long2 = int(4 * (self.long + 2 * self.area_adjust))
                img_crop = result[lat1:lat2, :, :][:, long1:long2, :]
                img_crop = self.upscaled_image(self.sr, img_crop)
                img_crop = np.concatenate(
                    [np.expand_dims(cv2.resize(img_crop[:, :, i],
                                               dsize=[128, 128],
                                               interpolation=cv2.INTER_CUBIC), axis=-1)
                     for i in range(img_crop.shape[-1])], -1)
            image_res.append(img_crop)
        image_res = np.array(image_res)
        return image_res

    def increase_resolution(self, images):
        image_res = []
        for i in range(images.shape[0]):
            image_res.append(self.super_resolution(images[i]))
        image_res = np.array(image_res)
        return image_res

    def get_image(self, dates):
        try:
            image_data = []
            for date_ in dates:
                file_sat = os.path.join(self.path_sat, 'processed', f'satellite_{date_.strftime("%Y_%m_%d__%H_%M")}.pkl')
                image_data.append(joblib.load(file_sat))
        except Exception as e:
            # print('Cannot find images for date {}'.format(date))
            raise e
        try:
            image_data = {k: np.concatenate([i[k] for i in image_data], axis=0) for k in image_data[0].keys()}

            spatial_coord_3d = torch.from_numpy(self.spatial_coords[:, :, :,
                                            self.lat - self.area_adjust:self.lat + self.area_adjust,
                                            self.long - self.area_adjust:self.long + self.area_adjust])
            x_img1 = []
            for img_tag in self.image_type:
                img_var = img_tag.split('_')[0]
                if 'coord' in img_var:
                    img = spatial_coord_3d
                else:
                    img = image_data[img_var]
                    img = self.increase_resolution(img)
                    if 'opt' in img_tag:
                        img_batch = rearrange(img[0], 't w h c -> t c w h')
                        img_batch = self.get_optical_flow(img_batch)
                        img_batch  = rearrange(img_batch, 't c w h -> t w h c')
                        img = np.expand_dims(img_batch, axis=0)
                    if 'grey' in img_tag:
                        img = self.get_image_grey(img)
                    img = torch.from_numpy(rearrange(img[:, :,
                                            self.lat - self.area_adjust:self.lat + self.area_adjust,
                                            self.long - self.area_adjust:self.long + self.area_adjust, :]
                                            , 'b t w h c -> b t c w h'))
                x_img1.append(img)

            x_img = torch.cat(x_img1, 2)
        except:
            print('Something went wrong')
            raise
        return x_img


    def get(self, date):
        dates_obs = pd.DatetimeIndex([date + pd.DateOffset(hours=l) for l in self.lags][::-1])
        dates_pred = pd.date_range(date, date + pd.DateOffset(hours=self.horizon), freq=self.ts_resolution)
        try:
            x_img_obs = self.get_image(dates_obs)
        except:
            print('Something went wrong')
            raise
        return_tensors = {
            "date": date,
            "images": x_img_obs.float(),
        }
        if self.use_target:
            try:
                x_img_pred = self.get_image(dates_pred)
            except:
                print('Something went wrong')
                raise
            return_tensors["target"] = torch.from_numpy(x_img_pred).float()
        return return_tensors
