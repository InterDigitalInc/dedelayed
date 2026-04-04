# Copyright (c) 2026, InterDigital
# All rights reserved.
# See LICENSE under the root folder.

from __future__ import annotations

from abc import ABC, abstractmethod

import torch.nn as nn
from torch import Tensor


class Dedelayed_v1_Remote(nn.Module, ABC):
    @abstractmethod
    def forward(
        self,
        x_remote: Tensor | None = None,
        *,
        z_remote: Tensor | None = None,
        x_local_size: tuple[int, int],
        past_ticks: Tensor,
    ) -> dict[str, Tensor]:
        raise NotImplementedError


class Dedelayed_v1_Local(nn.Module, ABC):
    @abstractmethod
    def forward(
        self,
        x_local: Tensor,
        *,
        downlink_features: Tensor | None = None,
    ) -> dict[str, Tensor]:
        raise NotImplementedError


class Dedelayed_v1_Fused(nn.Module, ABC):
    remote_model: Dedelayed_v1_Remote
    local_model: Dedelayed_v1_Local

    @abstractmethod
    def forward(
        self,
        x_local: Tensor,
        x_remote: Tensor,
        past_ticks: Tensor,
    ) -> dict[str, Tensor]:
        raise NotImplementedError
