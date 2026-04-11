# Copyright (c) 2026, InterDigital
# All rights reserved.
# See LICENSE under the root folder.

from __future__ import annotations

from typing import NamedTuple

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
