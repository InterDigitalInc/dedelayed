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
        x_remote: Tensor,
        *,
        past_ticks: Tensor,
        x_local_size: tuple[int, int],
        output_keys: Sequence[str] = ("downlink_features",),
    ) -> dict[str, Tensor]:
        raise NotImplementedError

    @abstractmethod
    def image_only(self, x_remote_latest: Tensor) -> dict[str, Tensor]:
        raise NotImplementedError

    @abstractmethod
    def stream_init(self, x_remote_latest: Tensor) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def stream_step(
        self,
        x_remote_latest: Tensor,
        stream_state: Tensor,
        *,
        past_ticks: Tensor,
        x_local_size: tuple[int, int],
        output_keys: Sequence[str] = ("downlink_features",),
    ) -> tuple[dict[str, Tensor], Tensor]:
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

    @abstractmethod
    def downlink_features_shape(
        self, x_local_size: tuple[int, int]
    ) -> tuple[int, int, int]:
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
