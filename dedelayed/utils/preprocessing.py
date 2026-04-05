# Copyright (c) 2026, InterDigital
# All rights reserved.
# See LICENSE under the root folder.

from __future__ import annotations

import io

import PIL.Image
import torch


def compute_size(h: int, aspect: float, div: int) -> tuple[int, int]:
    h = int(h)
    w = int(aspect * h)
    return (h // div * div), (w // div * div)


def normalize_uint8(x: torch.Tensor) -> torch.Tensor:
    return x / 255.0


def compress_decompress(
    frame: PIL.Image.Image, compression: dict | None
) -> PIL.Image.Image:
    if compression is None:
        return frame.copy()

    with io.BytesIO() as buf:
        frame.save(
            buf,
            format=compression["format"],
            quality=compression["quality"],
            lossless=compression["lossless"],
        )
        buf.seek(0)
        with PIL.Image.open(buf) as img:
            img.load()
            return img.copy()
