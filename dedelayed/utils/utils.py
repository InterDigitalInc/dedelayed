# Copyright (c) 2026, InterDigital
# All rights reserved.
# See LICENSE under the root folder.

from dataclasses import fields, is_dataclass, replace
from typing import Any, Callable, Self, TypeVar, cast

from torch import Tensor

T = TypeVar("T")
R = TypeVar("R")


def cache_by_id(func: Callable[[T], R]) -> Callable[[T], R]:
    cache: dict[int, R] = {}

    def cached(value: T) -> R:
        cache_key = id(value)
        if cache_key not in cache:
            cache[cache_key] = func(value)
        return cache[cache_key]

    return cached


def get_attr_by_key(obj: Any, key: str) -> Any:
    for subkey in filter(None, key.split(".")):
        obj = getattr(obj, subkey)
    return obj


class TensorContainerMixin:
    def to(self, *args: Any, **kwargs: Any) -> Self:
        return self._map_tensors(lambda x: x.to(*args, **kwargs))

    def pin_memory(self, *args: Any, **kwargs: Any) -> Self:
        return self._map_tensors(lambda x: x.pin_memory(*args, **kwargs))

    def _map_tensors(self, func: Callable[[Tensor], Tensor]) -> Self:
        def map_node(node: object) -> object:
            if isinstance(node, Tensor):
                return func(node)
            if isinstance(node, list):
                mapped_items = [map_node(item) for item in node]
                return type(node)(mapped_items)
            if is_dataclass(node) and not isinstance(node, type):
                mapped_field_values = {
                    field.name: map_node(getattr(node, field.name))
                    for field in fields(node)
                }
                return replace(node, **mapped_field_values)
            return node

        return cast(Self, map_node(self))
