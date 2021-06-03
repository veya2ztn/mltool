# copy from https://github.com/liamcli/gaea_release
from torch.optim.lr_scheduler import _LRScheduler,ReduceLROnPlateau
import numpy as np

class TriangleScheduler(_LRScheduler):
    """
    Simple class to linearly increase lr until loss stops decreasing
    with a certain grace period, then, linearly decrease lr.
    """
    def __init__(
        self,
        optimizer,
        slope=0.003,
        patience=5,
        max_epoch=30,
        last_epoch=-1,
    ):
        self.optimizer=optimizer
        if last_epoch == -1:
            self.min_lr = self.optimizer.param_groups[0]["lr"]
        self.patience = patience
        self.slope = slope
        self.counter = 0
        self.min_loss = 100
        self.increase = True
        self.last_epoch=last_epoch
        self.max_epoch = max_epoch
        self.base_lrs = [group['lr'] for group in self.optimizer.param_groups]

    def state_dict(self):
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)

    def get_lr(self):
        if self.last_epoch == 0:return self.base_lrs

        lrs = []
        for pg in self.optimizer.param_groups:
            lr = pg["lr"]
            lr = lr + self.slope if self.increase else lr - (self.counter>3)*self.slope
            lr = max(lr, self.min_lr)
            lrs.append(lr)
        return lrs

    def update_lr_state(self, loss):
        """
        Toggles loss to decreasing if grace period before loss decrease
        is used up.
        """

        if (self.min_loss is None) or (loss < self.min_loss):
            self.min_loss = loss
            self.counter  = 0
        else:
            self.counter += 1

        if ((self.counter > self.patience) or (self.last_epoch>self.max_epoch))and (self.increase):
            self.increase = False
            self.counter  = 0

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
        power = 1 ,
        cycles= 1,
        max_epoch = 20,
        cycle_decay=0.5,
        last_epoch=-1,
        min_lr=0.0001,
    ):
        self.min_lr = min_lr
        self.power = power
        self.cycles = cycles
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

    def step(self, loss=None,epoch=None):
        # convert `metrics` to float, in case it's a zero-dim Tensor
        if epoch is None:epoch = self.last_epoch + 1
        self.last_epoch = epoch
        new_lrs = self.get_lr()
        for new_lr, param_group in zip(new_lrs,self.optimizer.param_groups):
            param_group['lr'] = new_lr

class LinearUpThenTriPower(_LRScheduler):
    def __init__(
        self,
        optimizer,
        power = 1 ,
        cycles= 1,
        max_epoch = 20,
        cycle_decay=0.8,
        last_epoch=-1,
        slope=0.003,
        patience=5,
        trifunc='cos',
    ):
        self.optimizer=optimizer
        if last_epoch == -1:
            self.min_lr = self.optimizer.param_groups[0]["lr"]
        self.power = power
        self.cycles = cycles
        self.cycle_decay = cycle_decay
        self.max_epoch = max_epoch
        self.epochs_per_cycle = int(self.max_epoch / self.cycles)
        self.last_epoch=last_epoch
        self.patience = patience
        self.slope = slope
        self.counter = 0
        self.min_loss = None
        self.increase = True
        self.trifunc  = np.cos if trifunc=='cos' else np.sin
        #super().__init__(optimizer, last_epoch)


    def get_lr(self):
        if self.increase:
            lrs = []
            for pg in self.optimizer.param_groups:
                lr = pg["lr"]
                lr = lr + (self.counter+1)*self.slope
                lr = max(lr, self.min_lr)
                lrs.append(lr)
            return lrs
        else:
            last_epoch = self.last_epoch-self.start_epoch
            cycle = int(last_epoch / self.epochs_per_cycle)
            lr_decay = self.cycle_decay ** cycle
            if self.power == 1:
                numerator = 0.5 * (1+ self.trifunc(np.pi* (last_epoch% self.epochs_per_cycle)/ self.epochs_per_cycle))
                denominator = 1
            else:
                numerator = (self.power**(0.5* (1+ self.trifunc(np.pi* (last_epoch% self.epochs_per_cycle)/ self.epochs_per_cycle))+ 1)- self.power)
                denominator = self.power ** 2 - self.power

            return [
                self.min_lr + (lr_decay * base_lr - self.min_lr) * numerator / denominator
                for base_lr in self.base_lrs
            ]

    def update_lr_state(self, loss):
        """
        Toggles loss to decreasing if grace period before loss decrease
        is used up.
        """

        if (self.min_loss is None) or (loss < self.min_loss):
            self.min_loss = loss
            self.counter  = 0
        else:
            self.counter += 1

        if ((self.counter > self.patience) or (self.last_epoch>self.max_epoch))and (self.increase):
            self.increase = False
            self.counter  = 0
            self.base_lrs = [group['lr'] for group in self.optimizer.param_groups]
            self.start_epoch = self.last_epoch

    def step(self, loss, epoch=None):
        # convert `metrics` to float, in case it's a zero-dim Tensor
        self.last_epoch  = self.last_epoch + 1 if epoch is None else epoch
        self.update_lr_state(loss)
        new_lrs = self.get_lr()
        for new_lr, param_group in zip(new_lrs,self.optimizer.param_groups):
            param_group['lr'] = new_lr
