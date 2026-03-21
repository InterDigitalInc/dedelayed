# Copyright (c) 2026, InterDigital
# All rights reserved.
# See LICENSE under the root folder.

from __future__ import annotations

import datasets
import PIL.Image


def load_dataset(*args, **kwargs) -> datasets.Dataset:
    ds = datasets.load_dataset(*args, **kwargs)
    assert isinstance(ds, datasets.Dataset)
    features = {
        k: datasets.features.Image(decode=False)  # Avoid auto-decoding.
        if isinstance(v, datasets.features.Image)
        else v
        for k, v in ds.features.items()
    }
    return ds.cast(datasets.Features(features))


def decode_image(value) -> PIL.Image.Image:
    if isinstance(value, PIL.Image.Image):
        return value
    return datasets.Image(decode=True).decode_example(value)
