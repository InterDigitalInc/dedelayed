# Copyright (c) 2026, InterDigital
# All rights reserved.
# See LICENSE under the root folder.

from typing import Callable, TypeVar

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
