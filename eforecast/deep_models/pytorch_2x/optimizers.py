import torch
import torch.nn as nn
import torch.optim as opt


def get_rated(rated, y):
    if rated is not None:
        norm_val = torch.Tensor([1]).squeeze()
    else:
        norm_val = y
    return norm_val


class ReduceLROnPlateau:
    def __init__(self, optimizer, mode='min', step_reset_lr=50, scale_reset_lr=0.8, factor=0.8, patience=10,
                 threshold=1e-4, threshold_mode='rel', cooldown=0,
                 min_lr=0, eps=1e-8, verbose=False):

        if factor >= 1.0:
            raise ValueError('Factor should be < 1.0.')
        self.factor = factor
        self.step_reset_lr = step_reset_lr
        self.scale_reset_lr = scale_reset_lr
        # Attach optimizer
        if not isinstance(optimizer, opt.Optimizer):
            raise TypeError(f'{type(optimizer).__name__} is not an Optimizer')
        self.optimizer = optimizer
        self.init_lr = optimizer.param_groups[0]['lr']

        if isinstance(min_lr, (list, tuple)):
            if len(min_lr) != len(optimizer.param_groups):
                raise ValueError(f"expected {len(optimizer.param_groups)} min_lrs, got {len(min_lr)}")
            self.min_lrs = list(min_lr)
        else:
            self.min_lrs = [min_lr] * len(optimizer.param_groups)

        self.patience = patience
        self.verbose = verbose
        self.cooldown = cooldown
        self.cooldown_counter = 0
        self.mode = mode
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.best = None
        self.num_bad_epochs = None
        self.mode_worse = None  # the worse value for the chosen mode
        self.eps = eps
        self.last_epoch = 0
        self._init_is_better(mode=mode, threshold=threshold,
                             threshold_mode=threshold_mode)
        self._reset()

    def _reset(self):
        """Resets num_bad_epochs counter and cooldown counter."""
        self.best = self.mode_worse
        self.cooldown_counter = 0
        self.num_bad_epochs = 0

    def step(self, metrics, epoch=None):
        # convert `metrics` to float, in case it's a zero-dim Tensor
        current = float(metrics)
        if epoch is None:
            epoch = self.last_epoch + 1

        self.last_epoch = epoch
        if (self.last_epoch % self.step_reset_lr) == 0:
            self._init_lr(epoch)
        if self.is_better(current, self.best):
            self.best = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.in_cooldown:
            self.cooldown_counter -= 1
            self.num_bad_epochs = 0  # ignore any bad epochs in cooldown

        if self.num_bad_epochs > self.patience:
            self._reduce_lr(epoch)
            self.cooldown_counter = self.cooldown
            self.num_bad_epochs = 0

        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]

    def _reduce_lr(self, epoch):
        for i, param_group in enumerate(self.optimizer.param_groups):
            old_lr = float(param_group['lr'])
            new_lr = max(old_lr * self.factor, self.min_lrs[i])
            if old_lr - new_lr > self.eps:
                param_group['lr'] = new_lr
                if self.verbose:
                    epoch_str = ("%.2f" if isinstance(epoch, float) else
                                 "%.5d") % epoch
                    print(f'Epoch {epoch_str}: reducing learning rate of group {i} to {new_lr:.4e}.')

    def _init_lr(self, epoch):
        for i, param_group in enumerate(self.optimizer.param_groups):
            self.init_lr *= self.scale_reset_lr
            param_group['lr'] = max(self.init_lr, param_group['lr'])
            if self.verbose:
                epoch_str = ("%.2f" if isinstance(epoch, float) else
                             "%.5d") % epoch
                print(f'Epoch {epoch_str}: reducing learning rate of group {i} to {self.init_lr:.4e}.')

    @property
    def in_cooldown(self):
        return self.cooldown_counter > 0

    def is_better(self, a, best):
        if self.mode == 'min' and self.threshold_mode == 'rel':
            rel_epsilon = 1. - self.threshold
            return a < best * rel_epsilon

        elif self.mode == 'min' and self.threshold_mode == 'abs':
            return a < best - self.threshold

        elif self.mode == 'max' and self.threshold_mode == 'rel':
            rel_epsilon = self.threshold + 1.
            return a > best * rel_epsilon

        else:  # mode == 'max' and epsilon_mode == 'abs':
            return a > best + self.threshold

    def _init_is_better(self, mode, threshold, threshold_mode):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if threshold_mode not in {'rel', 'abs'}:
            raise ValueError('threshold mode ' + threshold_mode + ' is unknown!')

        if mode == 'min':
            self.mode_worse = torch.inf
        else:  # mode == 'max':
            self.mode_worse = -torch.inf

        self.mode = mode
        self.threshold = threshold
        self.threshold_mode = threshold_mode

    def state_dict(self):
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)
        self._init_is_better(mode=self.mode, threshold=self.threshold, threshold_mode=self.threshold_mode)


def triangular1_1_scale_fn(x):
    return 1 / (1.2 ** (x - 1))


class MeanAbsolutePercentageError(nn.Module):
    def __init__(self, rated):
        self.rated = rated
        super(MeanAbsolutePercentageError, self).__init__()

    def forward(self, model_output, y):
        rated = get_rated(self.rated, y)
        return torch.mean(torch.abs(torch.div(model_output - y, rated)))


class proba_loss(nn.Module):
    def __init__(self, quantiles=None, summation=True):
        super(proba_loss, self).__init__()
        self.quantiles = quantiles
        self.summation = summation

    def forward(self, model_output, y):
        losses = []
        for i, q in enumerate(self.quantiles):
            error = y - model_output[i]
            loss = torch.mean(torch.maximum(q * error, (q - 1) * error),
                              -1)

            losses.append(loss)
        if self.summation:
            return torch.stack(losses, dim=1).sum(dim=1).sum()
        else:
            return torch.stack(losses, dim=1).sum(dim=1).mean()


def optimize(net_model, device, optimizer='adam',
             scheduler='CosineAnnealing', rated=None,
             learning_rate=1e-4, is_fuzzy=False,
             probabilistic=False, quantiles=None):
    optimizers = dict()
    schedulers = dict()
    if is_fuzzy:
        params = [v for name, v in net_model.named_parameters()  if 'RBF_variance' in name]
        optimizers['fuzzy'] = opt.Adam(params, lr=10 * learning_rate)
        schedulers['fuzzy'] = torch.optim.lr_scheduler.StepLR(optimizers['fuzzy'], step_size=50, gamma=0.9)
        params = [v for name, v in net_model.named_parameters() if 'RBF_variance' not in name]
        optimizers['output'] = opt.Adam(params, lr=learning_rate, eps=learning_rate / 10)
        schedulers['output'] = ReduceLROnPlateau(optimizers['output'], 'min')

    if optimizer == 'adam':
        optimizers['bulk'] = opt.Adam(net_model.parameters(), lr=learning_rate)
    elif optimizer == 'adamw':
        optimizers['bulk'] = opt.AdamW(net_model.parameters(), lr=learning_rate)
    else:
        raise ValueError(f'optimizer {optimizer} not supported')
    if scheduler == 'CosineAnnealing':
        schedulers['bulk'] = opt.lr_scheduler.CosineAnnealingWarmRestarts(optimizers['bulk'],
                                                                          10, T_mult=2,
                                                                          eta_min=learning_rate / 100)
    elif scheduler == 'ReduceLROnPlateau':
        schedulers['bulk'] = ReduceLROnPlateau(optimizers['bulk'], 'min')
    else:
        raise ValueError(f'scheduler {scheduler} not supported')

    if probabilistic:
        loss = proba_loss(quantiles=quantiles).to(device)
        accuracy_out = proba_loss(quantiles=quantiles, summation=False).to(device)
        sse_out = proba_loss(quantiles=quantiles).to(device)
    else:
        loss = nn.MSELoss(reduction='sum')
        if rated is not None:
            accuracy_out = nn.L1Loss().to(device)
            sse_out = nn.MSELoss(reduction='mean').to(device)
        else:
            accuracy_out = MeanAbsolutePercentageError(rated).to(device)
            sse_out = nn.MSELoss(reduction='mean').to(device)
    return optimizers, schedulers, loss, accuracy_out, sse_out
