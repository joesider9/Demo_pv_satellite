import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


def feed_data_eval(data, ind=None, device=None):
    x = dict()
    if isinstance(data, dict):
        for name in data.keys():
            if isinstance(data[name], dict):
                x[name] = dict()
                for name1 in data[name].keys():
                    if ind is None:
                        x[name][name1] = torch.from_numpy(data[name][name1]).float()
                    else:
                        x[name][name1] = torch.from_numpy(data[name][name1][ind]).float()
                    if device is not None:
                        x[name][name1] = x[name][name1].to(device)
            else:
                if ind is None:
                    x[name] = torch.from_numpy(data[name]).float()
                else:
                    x[name] = torch.from_numpy(data[name][ind]).float()
                if device is not None:
                    x[name] = x[name].to(device)
    else:
        if ind is None:
            x['input'] = torch.from_numpy(data).float()
        else:
            x['input'] = torch.from_numpy(data[ind]).float()
        if device is not None:
            x['input'] = x['input'].to(device)
    return x


class DatasetDict(Dataset):
    def __init__(self, data, target):
        self.x = dict()
        if isinstance(data, dict):
            for name in data.keys():
                if isinstance(data[name], dict):
                    self.x[name] = dict()
                    for name1 in data[name].keys():
                        self.x[name][name1] = torch.from_numpy(data[name][name1])
                else:
                    self.x[name] = torch.from_numpy(data[name])
        else:
            self.x['input'] = torch.from_numpy(data)
        self.y = torch.from_numpy(target)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = dict()
        if isinstance(self.x, dict):
            for name in self.x.keys():
                if isinstance(self.x[name], dict):
                    x[name] = dict()
                    for name1 in self.x[name].keys():
                        x[name][name1] = self.x[name][name1][idx].float()
                else:
                    x[name] = self.x[name][idx].float()
        else:
            x['input'] = self.x[idx].float()
        y = self.y[idx].float()
        return x, y


def feed_dataset(data, target, batch_size=1, shuffle=False):
    # If use_image is True, then we create an image dataset which load image batch '.pt' and with image dates we select
    # the corresponding data. The length of the dataset is the number of image batches
    # Then we create a Dataloader with batch_size=1
    dataset = DatasetDict(data, target)
    dataloader = DataLoader(dataset, shuffle=shuffle, batch_size=batch_size)
    return dataloader


def train_(x_batch, y_batch, model, loss_fn, optimizer):
    # with torch.autograd.detect_anomaly(True):
    if 'row_data' in optimizer.keys():
        optimizer['row_data'].zero_grad()
        out_row_data = model(x_batch)[2]
        loss_out_row_data = loss_fn(out_row_data, y_batch.squeeze())
        loss_out_row_data.backward()
        optimizer['row_data'].step()

    if 'images' in optimizer.keys():
        optimizer['images'].zero_grad()
        out_images = model(x_batch)[1]
        loss_out_images = loss_fn(out_images, y_batch.squeeze())
        loss_out_images.backward()
        optimizer['images'].step()

    optimizer['bulk'].zero_grad()
    output = model(x_batch)[0]
    loss_out = loss_fn(output, y_batch.squeeze())
    loss_out.backward()
    optimizer['bulk'].step()



def move_to(obj, device):
    if torch.is_tensor(obj):
        return obj.to(device)
    elif isinstance(obj, pd.DatetimeIndex):
        pass
    elif isinstance(obj, dict):
        res = {}
        for k, v in obj.items():
            if k == 'dates':
                continue
            res[k] = move_to(v, device)
        return res
    elif isinstance(obj, list):
        res = []
        for v in obj:
            res.append(move_to(v, device))
        return res
    else:
        raise TypeError("Invalid type for move_to")


def train_step(model, loss_fn, optimizer, device, path_dataset, n_batch_train):
    model.train()
    start = time.time()
    for idx in tqdm(range(n_batch_train)):
        x_batch = torch.load(f"{path_dataset}/train_tensor{idx}.pt")
        x_batch = move_to(x_batch, device)
        y_batch = x_batch['target']
        train_(x_batch, y_batch, model, loss_fn, optimizer)
        end = time.time()
        if (end - start) > 300:
            break
    end = time.time()
    sec_per_iter = (end - start) / n_batch_train
    if sec_per_iter > 1:
        print(f'Run training step with {sec_per_iter}sec/iter')
    elif sec_per_iter > 0:
        print(f'Run training step with {1 / sec_per_iter}iter/sec')


def validation_step(model, performers, device, path_dataset, n_batch_val, mode='test'):
    model.eval()
    loss1 = 0
    loss2 = 0
    i = 0
    print('Validation')
    with torch.no_grad():
        for idx in tqdm(range(n_batch_val)):
            x_batch = torch.load(f"{path_dataset}/{mode}_tensor{idx}.pt")
            x_batch = move_to(x_batch, device)
            y_batch = x_batch['target']
            outputs = model(x_batch)[0]
            loss1 += (performers[0](outputs, y_batch.squeeze()))
            loss2 += (performers[1](outputs, y_batch.squeeze()))
            i += 1
    loss1 /= i
    loss2 /= i
    return loss1, loss2
