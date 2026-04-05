# Copyright (c) 2026, InterDigital
# All rights reserved.
# See LICENSE under the root folder.

from __future__ import annotations

import math

import torch


def get_raised_cosine_schedule(
    optimizer: torch.optim.Optimizer,
    *,
    num_training_steps: int,
    lr_pow: int = 2,
) -> torch.optim.lr_scheduler.LambdaLR:
    def lr_lambda(current_step: int) -> float:
        t = min(current_step / num_training_steps, 1.0)
        return 1.0 - ((math.cos(math.pi * t)) ** (2 * lr_pow))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
