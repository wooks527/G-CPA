from torch.optim.lr_scheduler import CosineAnnealingLR
import warnings
import math


class CosineAnnealingWithWarmUpLR(CosineAnnealingLR):
    """Set the learning rate using a cosine annealing schedule.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        T_max (int): Maximum number of iterations.
        eta_min (float): Minimum learning rate. Default: 0.
        last_epoch (int): The index of last epoch. Default: -1.
        warmup
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.
    """

    def __init__(
        self,
        optimizer,
        T_max,
        eta_min=1e-8,
        last_epoch=-1,
        warmup_epochs=5,
        verbose=False,
    ):
        self.T_max = T_max - warmup_epochs
        self.eta_min = eta_min
        self.warmup_epochs = warmup_epochs
        self.max_lr = optimizer.defaults["lr"]
        super(CosineAnnealingLR, self).__init__(optimizer, last_epoch, verbose)

        for param_group in optimizer.param_groups:
            param_group["lr"] = self.eta_min

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, "
                "please use `get_last_lr()`.",
                UserWarning,
            )

        if self.last_epoch == 0:
            return [group["lr"] for group in self.optimizer.param_groups]
        elif self._step_count == 1 and self.last_epoch > 0:
            cur_epoch = self.last_epoch - self.warmup_epochs
            return [
                self.eta_min
                + (base_lr - self.eta_min)
                * (1 + math.cos(cur_epoch * math.pi / self.T_max))
                / 2
                for base_lr, group in zip(
                    self.base_lrs,
                    self.optimizer.param_groups,
                )
            ]
        elif (self.last_epoch - self.warmup_epochs - 1 - self.T_max) % (
            2 * self.T_max
        ) == 0:
            cos_value = math.cos(math.pi / self.T_max)
            return [
                group["lr"] + (base_lr - self.eta_min) * (1 - cos_value) / 2
                for base_lr, group in zip(
                    self.base_lrs,
                    self.optimizer.param_groups,
                )
            ]
        elif self.last_epoch <= self.warmup_epochs:
            epoch_rate = self.last_epoch / self.warmup_epochs
            return [
                self.max_lr * epoch_rate + self.eta_min
                for _ in self.optimizer.param_groups
            ]
        cur_epoch = self.last_epoch - self.warmup_epochs
        return [
            (1 + math.cos(math.pi * cur_epoch / self.T_max))
            / (1 + math.cos(math.pi * (cur_epoch - 1) / self.T_max))
            * (group["lr"] - self.eta_min)
            + self.eta_min
            for group in self.optimizer.param_groups
        ]
