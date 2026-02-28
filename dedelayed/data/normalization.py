# Copyright (c) 2026, InterDigital
# All rights reserved.
# See LICENSE under the root folder.

from typing import Literal

import torch

ImageNormalizationKind = Literal["01", "minus1_1", "imagenet", "clip"]

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

_STATS = {
    "imagenet": (IMAGENET_MEAN, IMAGENET_STD),
    "clip": (CLIP_MEAN, CLIP_STD),
}


def renormalize(
    x: torch.Tensor,
    *,
    src: ImageNormalizationKind,
    dest: ImageNormalizationKind,
    channel_dim: int = -3,
) -> torch.Tensor:
    assert x.is_floating_point()

    if src == dest:
        return x

    channel_dim = channel_dim if channel_dim >= 0 else x.ndim + channel_dim
    assert 0 <= channel_dim < x.ndim

    def _mean_std(kind: ImageNormalizationKind) -> tuple[torch.Tensor, torch.Tensor]:
        mean_vals, std_vals = _STATS[kind]
        shape = [1] * x.ndim
        shape[channel_dim] = -1
        mean = x.new_tensor(mean_vals).view(*shape)
        std = x.new_tensor(std_vals).view(*shape)
        return mean, std

    if src == "01":
        x01 = x
    elif src == "minus1_1":
        x01 = (x + 1.0) * 0.5
    elif src in _STATS:
        mean, std = _mean_std(src)
        x01 = x * std + mean
    else:
        raise ValueError(f"Unknown src normalization: {src}")

    if dest == "01":
        return x01
    elif dest == "minus1_1":
        return x01 * 2.0 - 1.0
    elif dest in _STATS:
        mean, std = _mean_std(dest)
        return (x01 - mean) / std
    else:
        raise ValueError(f"Unknown dest normalization: {dest}")
