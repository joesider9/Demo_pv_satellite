import os
import joblib
import shutil
import copy
import time
import torch

import numpy as np
import pandas as pd

from eforecast.common_utils.train_utils import create_centroids
from eforecast.datasets.image_data.image_dataset import ImageDataset
from eforecast.datasets.image_data.image_dataset import ImageDataloader


def concat_pandas_df(df1, df2):
    dates = df1.index.intersection(df2.index)
    if len(dates) == 0:
        raise ValueError('Cannot sync datasets. there is no common dates')
    print(f'Merge pandas datasets with {dates.shape[0]} common dates ')
    return pd.concat([df1.loc[dates], df2.loc[dates]], axis=1)


def get_dates_from_dict(dict_df):
    dates = None
    for key, df in dict_df.items():
        if dates is None:
            dates = df.index
        else:
            dates = dates.intersection(df.index)
    return dates


def concat_df_dict(dict_df, df):
    dates = get_dates_from_dict(dict_df)
    dates = df.index.intersection(dates)
    dict_df_new = dict()
    for key, df_temp in dict_df.items():
        dict_df_new[key] = pd.concat([df.loc[dates], df_temp.loc[dates]], axis=1)
    print(f'Merge dictionary with pandas datasets ')
    return dict_df_new, dates

def concat_dict_dict(dict_df1, dict_df2):
    dates1 = get_dates_from_dict(dict_df1)
    dates2 = get_dates_from_dict(dict_df2)
    dates = dates1.intersection(dates2)
    if len(dates) == 0:
        raise ValueError('Cannot sync datasets. there is no common dates')
    dict_new = dict(zip(dates1, dates2))
    keys = set(list(dict_df1.keys()) + list(dict_df2.keys()))
    for key in keys:
        if key in dict_df1.keys() and key in dict_df2.keys():
            dict_new[key] = pd.concat([dict_df1[key].loc[dates], dict_df2[key].loc[dates]], axis=1)
        elif key in dict_df1.keys():
            dict_new[key] = dict_df1[key].loc[dates]
        else:
            dict_new[key] = dict_df2[key].loc[dates]
    print(f'Merge dictionary with dictionary found {dates.shape[0]} common dates ')
    return dict_new, dates


def concatenate_numpy(data1, dates1, data2, dates2):
    dates = dates1.intersection(dates2)
    data1 = data1[dates1.get_indexer(dates)]
    data2 = data2[dates2.get_indexer(dates)]
    return np.concatenate([data1, data2], axis=1)


def sync_data_tensor_with_tensors(data_tensor1=None, dates_tensor1=None, data_tensor2=None, dates_tensor2=None):
    dates = dates_tensor1.intersection(dates_tensor2)
    if isinstance(data_tensor1, dict):
        for key, data in data_tensor1.items():
            if not isinstance(data, np.ndarray):
                raise ValueError('data_tensor should be np.ndarray')
            data_tensor1[key] = data[dates_tensor1.get_indexer(dates)]
    else:
        data_tensor1 = data_tensor1[dates_tensor1.get_indexer(dates)]
    if isinstance(data_tensor2, dict):
        for key, data in data_tensor2.items():
            if not isinstance(data, np.ndarray):
                raise ValueError('data_tensor should be np.ndarray')
            data_tensor2[key] = data[dates_tensor2.get_indexer(dates)]
    else:
        data_tensor2 = data_tensor2[dates_tensor2.get_indexer(dates)]
    return data_tensor1, data_tensor2, dates


def sync_data_row_with_tensors(data_tensor=None, dates_tensor=None, data_row=None, dates_row=None):
    dates = dates_tensor.intersection(dates_row)
    if not isinstance(data_row, pd.DataFrame):
        raise ValueError('data_row should be dataframe')
    if isinstance(data_tensor, dict):
        for key, data in data_tensor.items():
            if not isinstance(data, np.ndarray):
                raise ValueError('data_tensor should be np.ndarray')
            data_tensor[key] = data[dates_tensor.get_indexer(dates)]
    else:
        data_tensor = data_tensor[dates_tensor.get_indexer(dates)]
    data_row = data_row.loc[dates]
    return data_tensor, data_row, dates


def np_ffill(arr, axis):
    idx_shape = tuple([slice(None)] + [np.newaxis] * (len(arr.shape) - axis - 1))
    idx = np.where(~np.isnan(arr), np.arange(arr.shape[axis])[idx_shape], 0)
    np.maximum.accumulate(idx, axis=axis, out=idx)
    slc = [np.arange(k)[tuple([slice(None) if dim==i else np.newaxis
        for dim in range(len(arr.shape))])]
        for i, k in enumerate(arr.shape)]
    slc[axis] = idx
    return arr[tuple(slc)]

def upsample_dict(data_dict, dates_dict, dates_new):
    if isinstance(data_dict, dict):
        for key, data in data_dict.items():
            data_dict[key] = upsample_dict(data, dates_dict, dates_new)
    elif isinstance(data_dict, np.ndarray):
        tensor_new = np.nan * np.ndarray([dates_new.shape[0]] + list(data_dict.shape[1:]))
        tensor_new[dates_new.get_indexer(dates_dict)] = data_dict
        tensor_new = np_ffill(tensor_new, 0)
        return tensor_new
    else:
        return upsample_dataset(data_dict, dates_new)


def upsample_tensor(data_tensor, dates_tensor):
    dates_new = pd.DatetimeIndex([])
    for date in dates_tensor:
        dates_new = dates_new.append(pd.date_range(date, periods=4, freq='15min'))

    data_tensor = upsample_dict(data_tensor, dates_tensor, dates_new)
    return data_tensor, dates_new

def downsample_dict(data_dict, dates_dict, resolution='D'):
    if isinstance(data_dict, dict):
        for key, data in data_dict.items():
            data_dict[key], dates_new = downsample_dict(data, dates_dict, resolution=resolution)
    elif isinstance(data_dict, np.ndarray):
        shape = data_dict.shape
        tensor_new = data_dict.reshape([shape[0], np.prod(shape[1:])])
        df = pd.DataFrame(tensor_new, index=dates_dict)
        df = df.resample(resolution).mean()
        return df.values.reshape(-1, *shape[1:]), df.index
    else:
        data_df = downsample_dataset(data_dict, resolution=resolution)
        return data_df, data_df.index
    return data_dict, dates_new


def downsample_tensor(data_tensor, dates_tensor, resolution='D'):
    data_tensor, dates_new = downsample_dict(data_tensor, dates_tensor, resolution=resolution)
    return data_tensor, dates_new

def upsample_dataset(df, dates):
    df_temp = pd.DataFrame(index=dates, columns=df.columns)
    df_temp.loc[df.index] = df
    df_temp = df_temp.bfill(axis=0, limit=1)
    df_temp = df_temp.ffill(axis=0, limit=2)
    return df_temp.dropna(axis=0)

def downsample_dataset(df, resolution='D'):
    return df.resample(resolution).mean()

def sync_target_with_tensors(target=None, data_tensor=None, dates_tensor=None, data_row=None):
    dates = dates_tensor.intersection(target.index)
    if isinstance(data_tensor, dict):
        for key, data in data_tensor.items():
            if isinstance(data, np.ndarray):
                data_tensor[key] = data[dates_tensor.get_indexer(dates)]
            else:
                data_tensor[key] = data.iloc[dates_tensor.get_indexer(dates)]
    else:
        data_tensor = data_tensor[dates_tensor.get_indexer(dates)]
    target = target.loc[dates]
    if data_row is not None:
        data_row = data_row.loc[dates]
        return [data_tensor, data_row], target
    else:
        return data_tensor, target

def sync_datasets(df1, df2):
    dates = df1.index.intersection(df2.index)
    if len(dates) == 0:
        raise ValueError('Cannot sync datasets. there is no common dates')
    print(f'Sync pandas datasets with {dates.shape[0]} common dates ')
    return df1.loc[dates], df2.loc[dates]

def find_common_dates(dict_df, dates_dict=None):
    dates = []
    if isinstance(dict_df, dict):
        for k, v in dict_df.items():
            results = find_common_dates(v, dates_dict=dates_dict)
            for res in results:
                dates.append(res)
    elif isinstance(dict_df, pd.DataFrame):
        dates.append(dict_df.index)
    elif isinstance(dict_df, np.ndarray):
        if dates_dict is None:
            raise ValueError('Data is not a dataframe. The corresponding dates are needed')
        dates.append(dates_dict)
    return dates

def get_data_dict_from_dates(dict_df, dates, dates_dict=None):
    if isinstance(dict_df, dict):
        for k, v in dict_df.items():
            dict_df[k] = get_data_dict_from_dates(v, dates, dates_dict=dates_dict)
    elif isinstance(dict_df, pd.DataFrame):
        return dict_df.loc[dates]
    elif isinstance(dict_df, np.ndarray):
        if dates_dict is None:
            raise ValueError('Data is not a dataframe. The corresponding dates are needed')
        ind = dates_dict.get_indexer(dates)
        return dict_df[ind]
    return dict_df


def sync_dict_df(dict_df, df2, dates_dict=None):
    dates_list = find_common_dates(dict_df, dates_dict=dates_dict)
    dates = pd.DatetimeIndex([])
    for date in dates_list:
        if dates.shape[0] == 0:
            dates = date
        else:
            dates = dates.intersection(date)
    dates = dates.intersection(df2.index)
    df2 = df2.loc[dates]
    if len(dates) == 0:
        raise ValueError('Cannot sync datasets. there is no common dates')
    dict_df = get_data_dict_from_dates(dict_df, dates, dates_dict=dates_dict)
    print(f'Sync pandas datasets with {dates.shape[0]} common dates ')
    return dict_df, df2


def fix_timeseries_dates(df, freq='h'):
    df.index = df.index.round(freq)
    df = df[~df.index.duplicated(keep='last')]
    dates = pd.date_range(df.index[0], df.index[-1], freq=freq)
    df_out = pd.DataFrame(index=dates, columns=df.columns)
    dates_in = dates.intersection(df.index)
    df_out.loc[dates_in] = df
    return df_out


def get_slice(data, ind):
    if isinstance(data, dict):
        for key, value in data.items():
            data[key] = get_slice(value, ind)
    elif isinstance(data, pd.DataFrame) or isinstance(data, pd.Series):
        return data.iloc[ind]
    else:
        return data[ind]
    return data


def get_slice_with_dates(data, data_dates, dates):
    if isinstance(data, dict):
        for key, value in data.items():
            data[key] = get_slice_with_dates(value, data_dates, dates)
    elif isinstance(data, pd.DataFrame) or isinstance(data, pd.Series):
        return data.loc[dates]
    else:
        ind = data_dates.get_indexer(dates)
        return data[ind]
    return data


def sync_data_with_dates(x, dates, dates_x=None):
    if dates_x is not None:
        dates_new = dates_x.intersection(dates)
        ind = dates_x.get_indexer(dates_new)
        return x[ind]
    else:
        dates_new = x.index.intersection(dates)
        return x.loc[dates_new]


def recursive_copy(src, dest):
    """
    Copy each file from src dir to dest dir, including sub-directories.
    """
    for item in os.listdir(src):
        file_path = os.path.join(src, item)

        # if item is a file, copy it
        if os.path.isfile(file_path):
            shutil.copy(file_path, dest)

        # else if item is a folder, recurse
        elif os.path.isdir(file_path):
            new_dest = os.path.join(dest, item)
            os.mkdir(new_dest)
            recursive_copy(file_path, new_dest)


def create_image_batches(static_data, params, batch_size, gpu_id, use_target):
    sat_image_type = params['sat_image_type']
    cv_masks = joblib.load(os.path.join(params['cluster_dir'], 'cv_mask.pickle'))

    X_train = ImageDataset(static_data, cv_masks[0], params, use_target=use_target)
    X_val = ImageDataset(static_data, cv_masks[1], params, use_target=use_target)
    X_test = ImageDataset(static_data, cv_masks[2], params, use_target=use_target)

    n_batch = int(batch_size)

    train_dataset = ImageDataloader(X_train, n_batch, 10)
    val_dataset = ImageDataloader(X_val, n_batch, 10)
    test_dataset = ImageDataloader(X_test, n_batch, 10)
    path_dataset = os.path.join(static_data['path_image'], 'SAT_DATA', sat_image_type.replace(':', '_'), f'gpu_id_{gpu_id}')

    if not os.path.exists(path_dataset):
        os.makedirs(path_dataset)
    print('begin to create batches....')
    start = time.time()
    n_batch_train = 0
    real_length = 0
    for idx in range(train_dataset.n_batches):
        if not os.path.exists(f"{path_dataset}/train_tensor{idx}.pt"):
            x_batch = train_dataset.get_batch()
            real_length += x_batch["dates"].shape[0]
            torch.save(x_batch, f"{path_dataset}/train_tensor{idx}.pt")
            if not train_dataset.valid:
                break
            print(f'Train batch #{idx}: written')
        n_batch_train += 1
    end = time.time()
    sec_per_iter = (end - start)
    print(f'Time elapsed for training batches {sec_per_iter}')
    print(f'real_length {real_length}')
    start = time.time()
    real_length = 0
    n_batch_val = 0
    for idx in range(val_dataset.n_batches):
        if not os.path.exists(f"{path_dataset}/val_tensor{idx}.pt"):
            x_batch = val_dataset.get_batch()
            real_length += x_batch["dates"].shape[0]
            torch.save(x_batch, f"{path_dataset}/val_tensor{idx}.pt")
            if not val_dataset.valid:
                break
            print(f'Val batch #{idx}: written')
        n_batch_val += 1
    end = time.time()
    sec_per_iter = (end - start)
    print(f'Time elapsed for validation batches {sec_per_iter}')
    print(f'real_length {real_length}')
    start = time.time()
    n_batch_test = 0
    real_length = 0
    for idx in range(test_dataset.n_batches):
        if not os.path.exists(f"{path_dataset}/test_tensor{idx}.pt"):
            x_batch = test_dataset.get_batch()
            real_length += x_batch["dates"].shape[0]
            torch.save(x_batch, f"{path_dataset}/test_tensor{idx}.pt")
            if not test_dataset.valid:
                break
            print(f'Test batch #{idx}: written')
        n_batch_test += 1
    end = time.time()
    sec_per_iter = (end - start)
    print(f'Time elapsed for testing batches {sec_per_iter}')
    print(f'real_length {real_length}')
    print('Create graph....')
    params['x_sample'] = torch.load(f"{path_dataset}/train_tensor0.pt")
    return params

def load_data_shallow_models(data_feeder, data_types, data_tag, train, get_lstm_vars=False):
    X = dict()
    metadata = dict()
    dates = None
    merge, compress, scale_nwp_method, scale_row_method, fs_method = (data_types[data_tag].get('merge'),
                                                           data_types[data_tag].get('compress'),
                                                           data_types[data_tag].get('scale_nwp_method'),
                                                           data_types[data_tag].get('scale_row_method'),
                                                           data_types[data_tag].get(
                                                               'feature_selection_method'))
    x, meta_x = data_feeder.feed_inputs(data_tag,
                                        merge=merge, compress=compress,
                                        scale_nwp_method=scale_nwp_method,
                                        scale_row_method=scale_row_method,
                                        feature_selection_method=fs_method,
                                        get_lstm_vars=get_lstm_vars)
    X.update(x)
    metadata.update(meta_x)
    dates = meta_x[data_tag]['dates'] if dates is None else meta_x[data_tag]['dates'].intersection(dates)
    if train:
        y = data_feeder.feed_target()
        X, y = sync_dict_df(X, y, dates_dict=dates)
        for data_tag in metadata.keys():
            metadata[data_tag]['dates'] = y.index
        return X, y, metadata,
    else:
        return X, metadata

def get_fuzzy_variables(params, columns):
    fuzzy_variables = dict()
    for var_name in params['var_imp']:
        var_names = [c for c in columns if var_name.lower() in c.lower()]
        if var_name not in columns:
            if len(var_names) == 0:
                raise ValueError(f'Cannot find variables associated with {var_name}')
            fuzzy_variables[var_name] = var_names
        else:
            fuzzy_variables[var_name] = var_name
    return fuzzy_variables


def load_data_deep_models(data_feeder, data_types, model_layers, params, cluster, train, is_fuzzy, refit):
    X = dict()
    metadata = dict()
    for data_tag in model_layers.keys():
        if data_tag in data_types.keys():
            merge, compress, scale_nwp_method, scale_row_method, fs_method = (data_types[data_tag].get('merge'),
                                                                   data_types[data_tag].get('compress'),
                                                                   data_types[data_tag].get('scale_nwp_method'),
                                                                   data_types[data_tag].get('scale_row_method'),
                                                                   data_types[data_tag].get(
                                                                       'feature_selection_method'))
            if data_tag == 'images':
                continue
                ## TODO Fix Image outputs in load_data_deep_models for Image2image case
            else:
                x, meta_x = data_feeder.feed_inputs(data_tag,
                                                   merge=merge, compress=compress,
                                                   scale_nwp_method=scale_nwp_method,
                                                   scale_row_method=scale_row_method,
                                                   feature_selection_method=fs_method,
                                                   cluster=cluster,
                                                   transform_calendar=True
                                                                      if model_layers[data_tag][0][0] != 'transformer'
                                                                      else False)
            if 'cal_vars' in meta_x[data_tag].keys():
                params['cal_vars'] = meta_x[data_tag]['cal_vars']
            X.update(x)
            metadata.update(meta_x)
        elif data_tag == 'images' or 'output' in data_tag or data_tag == 'hidden_layer':
            continue
        else:
            print(f'Warning: Net brunch of {data_tag} input tag ignored.')
    if is_fuzzy:
        if 'row_all' not in X.keys():
            raise RuntimeError("Dataset with tag 'row_all' not found to inputs")
        x = X['row_all']
        fuzzy_variables = get_fuzzy_variables(params, list(x.columns))
        X_imp = None
        for var_name, var_cols in fuzzy_variables.items():
            if isinstance(var_cols, list):
                var_mean = x[var_cols].mean(axis=1).to_frame(var_name)
                X_imp =  var_mean if X_imp is None else pd.concat([X_imp, var_mean], axis=1)
            else:
                var_mean = x[var_cols].to_frame(var_name)
                X_imp = var_mean if X_imp is None else pd.concat([X_imp, var_mean], axis=1)
        X['clustering'] = X_imp
        metadata['clustering'] = {'dates': X_imp.index}
        if refit or ('rules' not in params.keys() and 'centroids' not in params.keys()):
            y = data_feeder.feed_target()
            params = create_centroids(X_imp, y, params)
        new_X = dict()
        new_metadata = dict()
        new_model_layers = dict()
        for rule in params['rules']:
            for data_tag in X.keys():
                if data_tag not in {'clustering'}:
                    new_X[f'{data_tag}/{rule}'] = X[data_tag]
                    new_metadata[f'{data_tag}/{rule}'] = metadata[data_tag]
                    new_model_layers[f'{data_tag}/{rule}'] = model_layers[data_tag]
                else:
                    new_X[data_tag] = X[data_tag]
                    new_metadata[data_tag] = metadata[data_tag]
        for layer in model_layers.keys():
            if 'hidden' in layer or 'output' in layer or layer == 'images':
                new_model_layers[layer] = model_layers[layer]
        X = copy.deepcopy(new_X)
        metadata = copy.deepcopy(new_metadata)
        model_layers = copy.deepcopy(new_model_layers)
    else:
        params['rules'] = None
    new_X = dict()
    new_metadata = dict()
    new_model_layers = dict()
    for data_tag, data in X.items():
        if data_tag not in {'clustering'}:
            if data_tag not in model_layers.keys():
                raise ValueError(f'Cannot find {data_tag} tag in network architecture')
            elif isinstance(data, dict) and data_tag not in {'lstm'}:
                for tag, values in data.items():
                    new_X[f'{data_tag}/{tag}'] = X[data_tag][tag]
                    new_metadata[f'{data_tag}/{tag}'] = metadata[data_tag]
                    new_model_layers[f'{data_tag}/{tag}'] = model_layers[data_tag]
            else:
                new_X[data_tag] = X[data_tag]
                new_metadata[data_tag] = metadata[data_tag]
                new_model_layers[data_tag] = model_layers[data_tag]
        else:
            new_X[data_tag] = X[data_tag]
            new_metadata[data_tag] = metadata[data_tag]
    for layer in model_layers.keys():
        if 'hidden' in layer or 'output' in layer or layer == 'images':
            new_model_layers[layer] = model_layers[layer]
    X = copy.deepcopy(new_X)
    metadata = copy.deepcopy(new_metadata)
    model_layers = copy.deepcopy(new_model_layers)
    dates = None
    for data_tag in metadata.keys():
        if data_tag not in {'images'}:
            dates = metadata[data_tag]['dates'] if dates is None else dates.intersection(metadata[data_tag]['dates'])
    if train:
        y = data_feeder.feed_target()
        X, y = sync_dict_df(X, y, dates_dict=dates)
        for data_tag in metadata.keys():
            metadata[data_tag]['dates'] = y.index
        return X, y, metadata, model_layers, params
    else:
        for data_tag in metadata.keys():
            if isinstance(X[data_tag], pd.DataFrame):
                X[data_tag] = X[data_tag].loc[dates]
            elif isinstance(X[data_tag],np.ndarray):
                indices = metadata[data_tag]['dates'].get_indexer(dates)
                X[data_tag] = X[data_tag][indices]
            else:
                for key in X[data_tag].keys():
                    if isinstance(X[data_tag][key], pd.DataFrame):
                        X[data_tag][key] = X[data_tag][key].loc[dates]
                    elif isinstance(X[data_tag][key], np.ndarray):
                        indices = metadata[data_tag]['dates'].get_indexer(dates)
                        X[data_tag][key] = X[data_tag][key][indices]
                    else:
                        raise ValueError(f'Unknown type of data: {type(X[data_tag])}')
            metadata[data_tag]['dates'] = dates
        return X, metadata, model_layers, params

def get_slice_for_nets(x_dict, metadata_dict, dates=None, y=None):
    for data_tag, metadata in metadata_dict.items():
        if dates is None:
            dates = metadata['dates']
        else:
            dates = metadata['dates'].intersection(dates)
    if y is not None:
        dates = dates.intersection(y.index)
        y_slice = y.loc[dates].values
    else:
        y_slice = None
    X_slice = dict()
    for data_tag, x in x_dict.items():
        X_slice[data_tag] = get_slice_with_dates(x, metadata_dict[data_tag]['dates'], dates)

    return X_slice, y_slice, dates
