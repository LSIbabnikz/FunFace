
import math

from torch.optim.lr_scheduler import _LRScheduler


class PolyScheduler(_LRScheduler):
    def __init__(self, optimizer, base_lr, max_steps, warmup_steps, power=2, last_epoch=-1):
        self.base_lr = base_lr
        self.warmup_lr_init = 0.0001
        self.max_steps: int = max_steps
        self.warmup_steps: int = warmup_steps
        self.power = power
        super(PolyScheduler, self).__init__(optimizer, -1)
        self.last_epoch = last_epoch

    def get_warmup_lr(self):
        #alpha = float(self.last_epoch) / float(self.warmup_steps)
        alpha = math.sin((float(self.last_epoch) / float(self.warmup_steps)) * (math.pi/2.))
        return [self.base_lr * alpha for _ in self.optimizer.param_groups]

    def get_lr(self):
        if self.last_epoch == -1:
            return [self.warmup_lr_init for _ in self.optimizer.param_groups]
        if self.last_epoch < self.warmup_steps:
            return self.get_warmup_lr()
        else:
            alpha = pow(
                1
                - float(self.last_epoch - self.warmup_steps)
                / float(self.max_steps - self.warmup_steps),
                self.power,
            )
            return [self.base_lr * alpha for _ in self.optimizer.param_groups]


class StepScheduler(_LRScheduler):
    def __init__(self, optimizer, base_lr, max_steps, warmup_steps, steps, theta=0.1, last_epoch=-1):
        self.base_lr = base_lr
        self.warmup_lr_init = 0.0001
        self.max_steps: int = max_steps
        self.warmup_steps: int = warmup_steps
        self.theta = theta
        self.factor = 1.
        self.steps = steps
        super(StepScheduler, self).__init__(optimizer, -1)
        self.last_epoch = last_epoch

    def get_warmup_lr(self):
        #alpha = float(self.last_epoch) / float(self.warmup_steps)
        alpha = math.sin((float(self.last_epoch) / float(self.warmup_steps)) * (math.pi/2.))
        return [self.base_lr * alpha for _ in self.optimizer.param_groups]

    def get_lr(self):
        if self.last_epoch == -1:
            return [self.warmup_lr_init for _ in self.optimizer.param_groups]
        if self.last_epoch < self.warmup_steps:
            return self.get_warmup_lr()
        else:
            if self.last_epoch in self.steps:
                self.factor *= self.theta
            return [self.base_lr * self.factor for _ in self.optimizer.param_groups]
