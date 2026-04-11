# Copyright (c) 2026, InterDigital
# All rights reserved.
# See LICENSE under the root folder.

from __future__ import annotations

from typing import cast

from omegaconf import DictConfig, OmegaConf
from torch.utils.data import Dataset

from dedelayed.registry import DATASETS


def build_dataset(dataset_cfg: DictConfig | dict) -> Dataset:
    dataset_dict = (
        cast(dict, OmegaConf.to_container(dataset_cfg, resolve=True))
        if isinstance(dataset_cfg, DictConfig)
        else dataset_cfg
    )
    return DATASETS[dataset_dict["name"]](**dataset_dict["kwargs"])
