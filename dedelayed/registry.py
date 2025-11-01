from typing import Callable, Dict, Type, TypeVar

import torch.nn as nn

TModel = nn.Module
TModel_b = TypeVar("TModel_b", bound=TModel)

MODELS: Dict[str, Callable[..., TModel]] = {}


def register_model(name: str):
    """Decorator for registering a model."""

    def decorator(cls: Type[TModel_b]) -> Type[TModel_b]:
        MODELS[name] = cls
        return cls

    return decorator
