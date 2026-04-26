# Copyright (c) 2026, InterDigital
# All rights reserved.
# See LICENSE under the root folder.

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

import torch.nn as nn
from torch import Tensor


class Dedelayed_v1_Remote(nn.Module, ABC):
    @abstractmethod
    def forward(
        self,
        x_remote: Tensor | None = None,
        *,
        z_encoded: Tensor | None = None,
        x_local_size: tuple[int, int],
        past_ticks: Tensor,
        output_keys: Sequence[str] = ("downlink_features",),
    ) -> dict[str, Tensor]:
        raise NotImplementedError

    @abstractmethod
    def encode_frames(self, x_remote: Tensor) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def blend(self, z_encoded: Tensor) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def prealign(self, z_blended: Tensor, past_ticks: Tensor) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def head(
        self,
        z_prealigned: Tensor,
        x_local_size: tuple[int, int],
        output_keys: Sequence[str] = ("downlink_features",),
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
        local_only: bool = False,
    ) -> dict[str, Tensor]:
        raise NotImplementedError
