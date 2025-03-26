import time
import numpy as np
import tensorflow as tf
import keract


def train_schedule_fuzzy(model, loss_fns, optimizer, batch_size, x, y, warm):
    if warm > 0:
        for s in range(warm):
            print(f'WARMING STEP {s}')
            start = time.time()

            variables = [v for v in model.trainable_variables if 'centroid' not in v.name
                         or 'RBF_variance' not in v.name]
            train_step(model, loss_fns, optimizer, batch_size.astype(np.int64),
                       x, y, variables)
            end = time.time()
            sec_per_iter = (end - start) / int(y.shape[0] / batch_size)
            if sec_per_iter > 1:
                print(f'Run training step with {sec_per_iter}sec/iter')
            elif sec_per_iter > 0:
                print(f'Run training step with {1 / sec_per_iter}iter/sec')

    print('TRAINING fuzzy')
    start = time.time()
    variables = [v for v in model.trainable_variables if 'centroid' in v.name
                 or 'RBF_variance' in v.name]
    train_step(model, loss_fns, optimizer, batch_size.astype(np.int64),
               x, y, variables)

    end = time.time()
    sec_per_iter = (end - start) / int(y.shape[0] / batch_size)
    if sec_per_iter > 1:
        print(f'Run training step with {sec_per_iter}sec/iter')
    elif sec_per_iter > 0:
        print(f'Run training step with {1 / sec_per_iter}iter/sec')

    for s in range(3):
        print(f'TRAINING STEP {s}')
        print('TRAINING non Fuzzy')
        start = time.time()
        variables = [v for v in model.trainable_variables if 'centroid' not in v.name
                     or 'RBF_variance' not in v.name]
        train_step(model, loss_fns, optimizer, batch_size.astype(np.int64),
                   x, y, variables)

        end = time.time()
        sec_per_iter = (end - start) / int(y.shape[0] / batch_size)
        if sec_per_iter > 1:
            print(f'Run training step with {sec_per_iter}sec/iter')
        elif sec_per_iter > 0:
            print(f'Run training step with {1 / sec_per_iter}iter/sec')


def train_schedule_global(model, loss_fns, optimizer, batch_size, x, y):
    print('TRAINING bulk')
    start = time.time()
    train_step(model, loss_fns, optimizer, batch_size.astype(np.int64),
               x, y, model.trainable_variables)

    end = time.time()
    sec_per_iter = (end - start) / int(y.shape[0] / batch_size)
    if sec_per_iter > 1:
        print(f'Run training step with {sec_per_iter}sec/iter')
    elif sec_per_iter > 0:
        print(f'Run training step with {1 / sec_per_iter}iter/sec')


def feed_data(batch, data, target=None):
    x = dict()
    if isinstance(data, dict):
        for name in data.keys():
            if isinstance(data[name], dict):
                x[name] = dict()
                for name1 in data[name].keys():
                    x[name][name1] = data[name][name1][batch]
            else:
                    x[name] = data[name][batch]
    else:
        x['input'] = data[batch]
    if target is not None:
        y = target[batch]
        return x, y
    else:
        return x


def feed_data_eval(data, target=None):
    x = dict()
    if isinstance(data, dict):
        for name in data.keys():
            if isinstance(data[name], dict):
                x[name] = dict()
                for name1 in data[name].keys():
                    x[name][name1] = data[name][name1]
            else:
                x[name] = data[name]
    else:
        x['input'] = data
    if target is not None:
        y = target
        return x, y
    else:
        return x


def feed_dataset(data, batch_size, target=None):
    x = dict()
    if isinstance(data, dict):
        for name in data.keys():
            if isinstance(data[name], dict):
                x[name] = dict()
                for name1 in data[name].keys():
                    x[name][name1] = data[name][name1]
            else:
                x[name] = data[name]
    else:
        x['input'] = data
    if target is not None:
        y = target
        dataset = tf.data.Dataset.from_tensor_slices((x, y))
        dataset = dataset.shuffle(y.shape[0]).batch(batch_size)
        return dataset
    else:
        return tf.data.Dataset.from_tensor_slices(x)


@tf.function
def train_(x_batch, y_batch, model, loss_fn, optimizer, variables):
    with tf.GradientTape() as tape:
        predictions = model(x_batch, training=True)
        pred_loss = loss_fn(y_batch, predictions)
    gradients = tape.gradient(pred_loss, variables)
    gradients = [(tf.clip_by_value(grad, -10.0, 10.0)) for grad in gradients]
    optimizer.apply_gradients(zip(gradients, variables))


def train_step(model, loss_fn, optimizer, batch_size, x, y, variables):
    start = time.time()
    dataset = feed_dataset(x, batch_size, target=y)
    for x_batch, y_batch in dataset:
        train_(x_batch, y_batch, model, loss_fn, optimizer, variables)
    end = time.time()
    sec_per_iter = (end - start) / len(dataset)
    if sec_per_iter > 1:
        print(f'Run training step with {sec_per_iter}sec/iter')
    elif sec_per_iter > 0:
        print(f'Run training step with {1 / sec_per_iter}iter/sec')

def validation_step(model, performer, x, y):
    performer.reset_states()
    x, y = feed_data(np.arange(y.shape[0]), x, y)
    pred = model(x)
    if isinstance(performer, tf.keras.metrics.MeanAbsolutePercentageError):
        return performer(y, pred) / 100
    else:
        return performer(y, pred)


def compute_tensors(model, tensor_name, ind_val_list, x):
    x_batch = feed_data(ind_val_list, x)
    return keract.get_activations(model, x_batch, layer_names=tensor_name)[tensor_name]


def evaluate_activations(model, ind_train, x, thres_act):
    act_all_eval = compute_tensors(model, 'activations', ind_train, x)
    act_all_eval[act_all_eval >= thres_act] = 1
    act_all_eval[act_all_eval < thres_act] = 0
    print(f'SHAPE OF ACTIVATIONS IS {act_all_eval.shape}')
    print(f'SUM OF ACTIVATIONS IS {act_all_eval.sum()}')
    print(f'MIN OF ACTIVATIONS IS {act_all_eval.sum(axis=0).min()}')
    print(f'MAX OF ACTIVATIONS IS {act_all_eval.sum(axis=0).max()}')
    print(f'MEAN OF ACTIVATIONS IS {act_all_eval.sum(axis=0).mean()}')
    return act_all_eval.sum(), act_all_eval.sum(axis=0).min(), act_all_eval.sum(axis=0).max(), \
        act_all_eval.sum(axis=0).mean(), act_all_eval.sum(axis=0).argmin(), act_all_eval.sum(axis=0).argmax()


def gather_weights(model):
    best_layers = dict()
    with tf.GradientTape() as tape:
        for variable in model.trainable_variables:
            best_layers[variable.name] = variable.numpy()
    return best_layers
