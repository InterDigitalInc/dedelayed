# Copyright (c) 2026, InterDigital
# All rights reserved.
# See LICENSE under the root folder.

from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import numpy as np
import PIL.Image
import torch
from torchvision import tv_tensors
from torchvision.transforms import v2 as T
from torchvision.transforms.v2 import Resize

from dedelayed.datasets.hf import decode_image
from dedelayed.utils.preprocessing import (
    compress_decompress,
    normalize_uint8,
)
from dedelayed.utils.utils import cache_by_id


def build_train_transform(
    source_size: tuple[int, int],
    crop_scale: tuple[float, float] = (0.65, 1.0),
) -> T.Compose:
    h, w = source_size
    fill: dict = {tv_tensors.Image: 0, tv_tensors.Mask: 255}
    return T.Compose(
        [
            T.RandomHorizontalFlip(0.5),
            T.RandomApply(
                [
                    T.RandomAffine(
                        degrees=[-7.0, 7.0],
                        shear=(-3.0, 3.0, -3.0, 3.0),
                        fill=fill,
                        interpolation=PIL.Image.Resampling.BILINEAR,
                    ),
                ],
                p=0.1,
            ),
            T.RandomResizedCrop(
                size=(h, w),
                scale=crop_scale,
                ratio=((w / h) * 0.75, (w / h) / 0.75),
                interpolation=PIL.Image.Resampling.BICUBIC,
            ),
            T.RandomApply(
                [T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)], p=0.3
            ),
            T.ToPureTensor(),
        ]
    )


def build_eval_transform() -> T.Compose:
    return T.Compose(
        [
            T.ToPureTensor(),
        ]
    )


class Clip(NamedTuple):
    x_remote: list[torch.Tensor]
    x_local: list[torch.Tensor]
    target: list[torch.Tensor]


class ClipIdx(NamedTuple):
    x_remote: list[int]
    x_local: list[int]
    target: list[int]

    def speedup(self, factor: int) -> ClipIdx:
        return ClipIdx(
            x_remote=[factor * i for i in self.x_remote],
            x_local=[factor * i for i in self.x_local],
            target=[factor * i for i in self.target],
        )

    def shift(self, offset: int) -> ClipIdx:
        return ClipIdx(
            x_remote=[i + offset for i in self.x_remote],
            x_local=[i + offset for i in self.x_local],
            target=[i + offset for i in self.target],
        )


class ComposeTemporal:
    def __init__(self, transforms: list[Callable[[ClipIdx], ClipIdx]]) -> None:
        self.transforms = transforms

    def __call__(self, idx: ClipIdx) -> ClipIdx:
        for transform in self.transforms:
            idx = transform(idx)
        return idx


class RandomSpeedupShift:
    def __init__(self, speedups: tuple[int, ...], idx_range: tuple[int, int]) -> None:
        self._speedups = tuple(int(s) for s in speedups)
        self._idx_range = tuple(int(i) for i in idx_range)

    def __call__(self, idx: ClipIdx) -> ClipIdx:
        idxs = [*idx.x_remote, *idx.x_local, *idx.target]
        min_idx = min(idxs)
        max_idx = max(idxs)
        lo, hi = self._idx_range
        speedups = [s for s in self._speedups if s * (max_idx - min_idx) <= hi - lo]
        speedup = int(np.random.choice(speedups))

        offset_min = lo - speedup * min_idx
        offset_max = hi - speedup * max_idx
        offset = int(np.random.choice(range(offset_min, offset_max + 1)))

        return idx.speedup(speedup).shift(offset)


def resolve_clip_idx(
    idx: ClipIdx,
    sample: dict,
    *,
    past_ticks_true: int,
    future_ticks_true: int = 0,
) -> ClipIdx:
    rel_idx_anchor = sample.get("rel_idx_anchor")
    if rel_idx_anchor is None:  # If unspecified, assume all frames are labeled.
        offset = 0
    else:
        anchor_in_canonical = {
            "remote_latest": -past_ticks_true,
            "local_latest": 0,
            "target": future_ticks_true,
        }[rel_idx_anchor]
        offset = -anchor_in_canonical
    return idx.shift(offset)


def preprocess_clip(
    sample: dict,
    idx: ClipIdx,
    *,
    uplink_compression: dict | None,
    transform: T.Transform,
    x_remote_size: tuple[int, int],
    x_local_size: tuple[int, int],
    interpolation: PIL.Image.Resampling = PIL.Image.Resampling.BICUBIC,
) -> Clip:
    assert isinstance(sample, dict)
    decode = cache_by_id(decode_image)

    x_remote_src = [decode(sample["remote_frame"][i]) for i in idx.x_remote]
    x_local_src = [decode(sample["local_frame"][i]) for i in idx.x_local]
    target_src = [decode(sample["seg_mask"][i]) for i in idx.target]

    x_remote_src = [
        compress_decompress(frame, uplink_compression) for frame in x_remote_src
    ]

    x_remote_i, x_local_i, target_i = transform(
        [tv_tensors.Image(frame) for frame in x_remote_src],
        [tv_tensors.Image(frame) for frame in x_local_src],
        [tv_tensors.Mask(frame) for frame in target_src],
    )

    x_remote_i = [
        normalize_uint8(Resize(x_remote_size, interpolation)(frame))
        for frame in x_remote_i
    ]
    x_local_i = [
        normalize_uint8(Resize(x_local_size, interpolation)(frame))
        for frame in x_local_i
    ]
    target_i = [frame.squeeze(0).to(torch.long) for frame in target_i]

    return Clip(x_remote_i, x_local_i, target_i)
