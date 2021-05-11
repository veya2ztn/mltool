# copy from https://github.com/liamcli/gaea_release
from torch.optim.lr_scheduler import _LRScheduler,ReduceLROnPlateau
import numpy as np

class TriangleScheduler(ReduceLROnPlateau):
    """
    Simple class to linearly increase lr until loss stops decreasing
    with a certain grace period, then, linearly decrease lr.
    """
    def __init__(
        self,
        optimizer,
        slope=0.003,
        max_lr=0.1,
        min_lr=0.001,
        patience=5,
        max_epoch=3000,
        last_epoch=-1,
    ):
        self.optimizer=optimizer
        self.slope = slope
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.patience = patience
        self.max_epoch = max_epoch
        self.counter = 0
        self.min_loss = 100
        self.increase = True
        self.last_epoch=-1
        self.base_lrs = [group['lr'] for group in self.optimizer.param_groups]
    def state_dict(self):
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)

    def get_lr(self):
        if self.last_epoch == 0:
            return self.base_lrs

        lrs = []
        for pg in self.optimizer.param_groups:
            lr = pg["lr"]
            if self.increase:
                lr += self.slope
            else:
                lr -= self.slope
            lr = max(min(lr, self.max_lr), self.min_lr)
            lrs.append(lr)
        return lrs

    def update_lr_state(self, loss):
        """
        Toggles loss to decreasing if grace period before loss decrease
        is used up.
        """
        if loss < self.min_loss - 0.01:self.min_loss = loss
        else:self.counter += 1
        if not self.increase and loss > self.min_loss:self.slope += self.slope
        if self.counter > self.patience:
            self.increase = False
        if (self.max_lr - self.min_lr) / (self.max_epoch - self.last_epoch + 0.01) > self.slope:
            self.increase = False

    def step(self, loss, epoch=None):
        # convert `metrics` to float, in case it's a zero-dim Tensor
        if epoch is None:
            epoch = self.last_epoch + 1
        else:
            warnings.warn(EPOCH_DEPRECATION_WARNING, UserWarning)
        self.last_epoch = epoch

        self.update_lr_state(loss)
        new_lrs = self.get_lr()
        for new_lr, param_group in zip(new_lrs,self.optimizer.param_groups):
            param_group['lr'] = new_lr

class CosinePowerAnnealing(_LRScheduler):
    def __init__(
        self,
        optimizer,
        max_epoch = 3000,
        cycles= 1,
        power = 1 ,
        min_lr= 0.0001,
        cycle_decay=0.5,
        last_epoch=-1,
    ):
        self.power = power
        self.cycles = cycles
        self.min_lr = min_lr
        self.cycle_decay = cycle_decay
        self.max_epoch = max_epoch
        self.epochs_per_cycle = int(self.max_epoch / self.cycles)
        super(CosinePowerAnnealing, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        cycle = int(self.last_epoch / self.epochs_per_cycle)
        lr_decay = self.cycle_decay ** cycle
        if self.power == 1:
            numerator = 0.5 * (
                1
                + np.cos(
                    np.pi
                    * (self.last_epoch % self.epochs_per_cycle)
                    / self.epochs_per_cycle
                )
            )
            denominator = 1
        else:
            numerator = (
                self.power
                ** (
                    0.5
                    * (
                        1
                        + np.cos(
                            np.pi
                            * (self.last_epoch % self.epochs_per_cycle)
                            / self.epochs_per_cycle
                        )
                    )
                    + 1
                )
                - self.power
            )
            denominator = self.power ** 2 - self.power

        return [
            self.min_lr + (lr_decay * base_lr - self.min_lr) * numerator / denominator
            for base_lr in self.base_lrs
        ]

    def step(self, epoch=None):
        # convert `metrics` to float, in case it's a zero-dim Tensor
        if epoch is None:epoch = self.last_epoch + 1
        self.last_epoch = epoch
        new_lrs = self.get_lr()
        for new_lr, param_group in zip(new_lrs,self.optimizer.param_groups):
            param_group['lr'] = new_lr
