# Copyright (c) 2026, InterDigital
# All rights reserved.
# See LICENSE under the root folder.

from __future__ import annotations

from typing import Type, TypeVar

import torch.nn as nn
from torch.utils.data import Dataset

TDataset = Dataset
TModel = nn.Module

TDataset_b = TypeVar("TDataset_b", bound=TDataset)
TModel_b = TypeVar("TModel_b", bound=TModel)

DATASETS: dict[str, Type[TDataset]] = {}
MODELS: dict[str, Type[TModel]] = {}


def register_dataset(name: str):
    """Decorator for registering a dataset."""

    def decorator(cls: Type[TDataset_b]) -> Type[TDataset_b]:
        DATASETS[name] = cls
        return cls

    return decorator


def register_model(name: str):
    """Decorator for registering a model."""

    def decorator(cls: Type[TModel_b]) -> Type[TModel_b]:
        MODELS[name] = cls
        return cls

    return decorator
