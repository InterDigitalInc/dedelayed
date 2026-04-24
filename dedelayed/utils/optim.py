# Copyright (c) 2026, InterDigital
# All rights reserved.
# See LICENSE under the root folder.

from __future__ import annotations

import math

import torch


class RaisedCosineLR(torch.optim.lr_scheduler.LambdaLR):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        *,
        num_training_steps: int,
        lr_pow: int = 2,
        min_lr_ratio: float = 1e-4,
        last_epoch: int = -1,
    ) -> None:
        self.num_training_steps = num_training_steps
        self.lr_pow = lr_pow
        self.min_lr_ratio = min_lr_ratio
        super().__init__(optimizer, self._lr_lambda, last_epoch=last_epoch)

    def _lr_lambda(self, current_step: int) -> float:
        t = min(current_step / self.num_training_steps, 1.0)
        scale = 1.0 - math.cos(math.pi * t) ** (2 * self.lr_pow)
        return self.min_lr_ratio + (1.0 - self.min_lr_ratio) * scale
