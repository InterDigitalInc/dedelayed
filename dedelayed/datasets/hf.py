# Copyright (c) 2026, InterDigital
# All rights reserved.
# See LICENSE under the root folder.

from __future__ import annotations

from typing import Any

import datasets
import PIL.Image
from torch.utils.data import Dataset

from dedelayed.registry import register_dataset


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


@register_dataset("hf")
class HfDataset(Dataset):
    def __init__(self, *, path: str, split: str) -> None:
        self._dataset = load_dataset(path, split=split)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self._dataset[idx]

    def __len__(self) -> int:
        return len(self._dataset)
