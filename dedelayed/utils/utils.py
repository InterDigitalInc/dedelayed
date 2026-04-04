# Copyright (c) 2026, InterDigital
# All rights reserved.
# See LICENSE under the root folder.

from typing import Any, Callable, TypeVar

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
