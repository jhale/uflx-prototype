# Copyright (C) 2026 Jack S. Hale
#
# This file is part of uflx-core-experimental (https://fenics.github.io/uflx/language/)
#
# SPDX-License-Identifier:    MIT
"""Per-node caching for computed properties."""

from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from typing import TypeVar, cast

_T = TypeVar("_T")
_R = TypeVar("_R")


def cached(fn: Callable[[_T], _R]) -> Callable[[_T], _R]:
    """Cache a property on its immutable node, without retaining nodes globally."""

    @wraps(fn)
    def read(node: _T) -> _R:
        values = vars(node)
        if fn.__name__ not in values:
            values[fn.__name__] = fn(node)
        return cast("_R", values[fn.__name__])

    return read
