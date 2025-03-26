import os
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from eforecast.deep_models.pytorch_2x.image_dataset_real_time import ImageDatasetRealTime


def train_schedule_fuzzy(model, loss, optimizer, dataset, device, warm):
    if warm > 0:
        for s in range(warm):
            print(f'WARMING STEP {s}')
            for name, v in model.named_parameters():
                if 'RBF_variance' in name:
                    v.requires_grad = False
            train_step(model, loss, optimizer['output'], dataset, device)

    print('TRAINING fuzzy')
    for name, v in model.named_parameters():
        if 'RBF_variance' in name:
            v.requires_grad = True

    train_step(model, loss, optimizer['fuzzy'], dataset, device)

    for name, v in model.named_parameters():
        if 'RBF_variance' in name:
            v.requires_grad = False

    for s in range(3):
        print(f'TRAINING STEP {s}')
        print('TRAINING non Fuzzy')
        train_step(model, loss, optimizer['output'], dataset, device)


def feed_data_eval(data):
    x = dict()
    if isinstance(data, dict):
        for name in data.keys():
            if isinstance(data[name], dict):
                x[name] = dict()
                for name1 in data[name].keys():
                    values = data[name][name1] if isinstance(data[name][name1], np.ndarray) \
                        else data[name][name1].values
                    x[name][name1] = torch.from_numpy(values).float()
            else:
                values = data[name] if isinstance(data[name], np.ndarray) else data[name].values
                x[name] = torch.from_numpy(values).float()
    return x


class DatasetDict(Dataset):
    def __init__(self, data, target, device):
        self.device = device
        self.x = dict()
        if isinstance(data, dict):
            for name in data.keys():
                if isinstance(data[name], dict):
                    self.x[name] = dict()
                    for name1 in data[name].keys():
                        values = data[name][name1] if isinstance(data[name][name1], np.ndarray) \
                            else data[name][name1].values
                        self.x[name][name1] = torch.from_numpy(values)
                else:
                    values = data[name] if isinstance(data[name], np.ndarray) else data[name].values
                    self.x[name] = torch.from_numpy(values)
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
                        x[name][name1] = self.x[name][name1][idx].float().to(self.device)
                else:
                    x[name] = self.x[name][idx].float().to(self.device)
        else:
            raise ValueError('Input must be dict')
        y = self.y[idx].float().to(self.device)
        return x, y


def collate_fn_eval(batch):
    fn = list(filter (lambda x:x[0] is not None if isinstance(x, tuple) else x is not None, batch))
    dict_batch = [item[0] for item in fn]
    timestamp_batch = [item[1] for item in fn]
    collated_dict = torch.utils.data.dataloader.default_collate(dict_batch)
    timestamp_tensor = torch.tensor([ts.timestamp() for ts in timestamp_batch])

    return collated_dict, timestamp_tensor

def collate_fn_train(batch):
    fn = list(filter (lambda x:x[0] is not None if isinstance(x, tuple) else x is not None, batch))
    return torch.utils.data.dataloader.default_collate(fn)

def feed_dataset(data, target, device, batch_size=1, shuffle=False):
    # If use_image is True, then we create an image dataset which load image batch '.pt' and with image dates we select
    # the corresponding data. The length of the dataset is the number of image batches
    # Then we create a Dataloader with batch_size=1
    dataset = DatasetDict(data, target, device)
    dataloader = DataLoader(dataset, shuffle=shuffle, batch_size=batch_size)
    return dataloader



class DatasetImageDict(Dataset):
    def __init__(self, data, target, dates, path_dataset, tag_dataset, n_batches, device, train=True):
        self.path_dataset = path_dataset
        self.tag_dataset = tag_dataset
        self.n_batches = n_batches
        self.dates = dates
        self.device = device
        self.x = dict()
        self.train = train
        if isinstance(data, dict):
            for name in data.keys():
                if name == 'images':
                    continue
                if isinstance(data[name], dict):
                    self.x[name] = dict()
                    for name1 in data[name].keys():
                        values = data[name][name1] if isinstance(data[name][name1], np.ndarray) \
                            else data[name][name1].values
                        self.x[name][name1] = torch.from_numpy(values)
                else:
                    values = data[name] if isinstance(data[name], np.ndarray) else data[name].values
                    self.x[name] = torch.from_numpy(values)
        else:
            self.x['input'] = torch.from_numpy(data)
        if train:
            self.y = torch.from_numpy(target) if target is not None else None
        else:
            self.y = None

    def __len__(self):
        return self.n_batches

    def __getitem__(self, idx):
        images = torch.load(f"{self.path_dataset}/{self.tag_dataset}_tensor{idx}.pt")
        if 'images' not in images.keys():
            return None, None
        dates_image = pd.DatetimeIndex([d[0] for d in images['dates']])
        dates_image_new = dates_image.intersection(self.dates)
        if dates_image_new.shape[0] == 0:
            return None, None
        indices = self.dates.get_indexer(dates_image_new)
        indices_image = dates_image.get_indexer(dates_image_new)
        x = dict()
        if isinstance(self.x, dict):
            for name in self.x.keys():
                if isinstance(self.x[name], dict):
                    x[name] = dict()
                    for name1 in self.x[name].keys():
                        x[name][name1] = self.x[name][name1][indices].float().to(self.device)
                else:
                    x[name] = self.x[name][indices].float().to(self.device)
        else:
            raise ValueError('Input must be dict')
        x['images'] = images['images'][indices_image].squeeze().float().to(self.device)
        if self.train:
            y = self.y[indices].float().to(self.device) if self.y is not None else images["target"].float().to(self.device)
            return x, y
        else:
            return x, dates_image_new


def feed_image_dataset(data, target, dates, tag_dataset, path_dataset, device, train=True):
    n_batches = len([p for p in os.listdir(path_dataset) if p.startswith(f"{tag_dataset}")])
    dataset = DatasetImageDict(data, target, dates, path_dataset, tag_dataset, n_batches, device, train=train)
    dataloader = DataLoader(dataset, batch_size=None)
    return dataloader


def feed_image_dataset_real_time(static_data, data, target, dates, params , device, batch_size=1, shuffle=False,
                                 train=True, use_target=True, is_online=False):
    dataset = ImageDatasetRealTime(static_data, data, target, dates, params, device, train=train, use_target=use_target,
                                   is_online=is_online)
    dataloader = DataLoader(dataset, shuffle=shuffle, batch_size=batch_size, collate_fn=collate_fn_train
                                                                                        if train else collate_fn_eval)
    return dataloader



def train_(x_batch, y_batch, model, loss_fn, optimizer):
    # with torch.autograd.detect_anomaly(True):
    outputs = model(x_batch)
    loss = loss_fn(outputs, y_batch)
    loss += model.act_nans
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

def train_step(model, loss_fn, optimizer, dataset, device):
    model.train()
    optimizer.zero_grad()
    start = time.time()
    for x_batch, y_batch in tqdm(dataset):
        if x_batch is None:
            continue
        train_(x_batch, y_batch, model, loss_fn, optimizer)
    end = time.time()
    sec_per_iter = (end - start) / len(dataset)
    if sec_per_iter > 1:
        print(f'Run training step with {sec_per_iter}sec/iter')
    elif sec_per_iter > 0:
        print(f'Run training step with {1 / sec_per_iter}iter/sec')


def validation_step(model, Accuracy, Sse, dataset, device, n_cvs=1):
    model.eval()
    loss1 = []
    loss2 = []
    with torch.no_grad():
        for x_batch, y_batch in tqdm(dataset):
            if x_batch is None:
                continue
            x_batch = x_batch
            y_batch = y_batch.to(device)
            outputs = model(x_batch)
            act_nans = model.act_nans
            loss1.append((Accuracy(outputs, y_batch) + act_nans).cpu().detach().numpy())
            loss2.append((Sse(outputs, y_batch) + act_nans).cpu().detach().numpy())
    if len(loss1) > n_cvs:
        loss1 = [np.mean(l) for l in np.array_split(loss1, n_cvs)]
        loss2 = [np.mean(l) for l in np.array_split(loss2, n_cvs)]
    return loss1, loss2


def compute_tensors(model, x):
    model.eval()
    return model(x, get_activations=True)


def evaluate_activations(model, x, thres_act):
    act_all_eval = compute_tensors(model, feed_data_eval(x))
    act_all_eval[act_all_eval >= thres_act] = 1
    act_all_eval[act_all_eval < thres_act] = 0
    print(f'SHAPE OF ACTIVATIONS IS {act_all_eval.shape}')
    print(f'SUM OF ACTIVATIONS IS {act_all_eval.sum()}')
    print(f'MIN OF ACTIVATIONS IS {act_all_eval.sum(axis=0).min()}')
    print(f'MAX OF ACTIVATIONS IS {act_all_eval.sum(axis=0).max()}')
    print(f'MEAN OF ACTIVATIONS IS {act_all_eval.sum(axis=0).mean()}')
    return act_all_eval.sum(), act_all_eval.sum(axis=0).min(), act_all_eval.sum(axis=0).max(), \
        act_all_eval.sum(axis=0).mean(), act_all_eval.sum(axis=0).argmin(), act_all_eval.sum(axis=0).argmax()