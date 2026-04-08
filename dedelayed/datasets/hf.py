# Copyright (c) 2026, InterDigital
# All rights reserved.
# See LICENSE under the root folder.

from __future__ import annotations

import re
from collections import defaultdict
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


@register_dataset("hf_temporal_columns")
class HfTemporalColumnsDataset(Dataset):
    def __init__(
        self,
        *,
        path: str,
        split: str,
        remap: dict[str, str],
        ref_idx: int = 0,
    ) -> None:
        self._dataset = load_dataset(path, split=split)
        self._remap = remap
        self._ref_idx = ref_idx

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return remap_and_gather_series(self._dataset[idx], self._remap, self._ref_idx)

    def __len__(self) -> int:
        return len(self._dataset)


def remap_and_gather_series(
    sample: dict[str, Any], remap: dict[str, str], ref_idx: int
) -> dict[str, Any]:
    remap_inv = defaultdict(list)
    for dst, src in remap.items():
        remap_inv[src].append(dst)
    pattern = re.compile(rf"^({'|'.join(map(re.escape, remap_inv))})_(\d+)$")
    grouped: dict[str, Any] = {dst: {} for dst in remap}
    for key, value in sample.items():
        match = pattern.fullmatch(key)
        if match is None:
            grouped[key] = value
            continue
        src, idx = match.groups()
        for dst in remap_inv[src]:
            grouped[dst][int(idx) - ref_idx] = value
    return grouped
