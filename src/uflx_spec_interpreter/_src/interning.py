from __future__ import annotations

import hashlib
import weakref
from collections.abc import Callable
from functools import wraps
from typing import TYPE_CHECKING, TypeVar, cast

if TYPE_CHECKING:
    from .nodes.base import Base
    from .types import _Construction


_T = TypeVar("_T")
_R = TypeVar("_R")
_table: weakref.WeakValueDictionary[tuple[type[Base], int], Base] = weakref.WeakValueDictionary()


# Shared implementation helpers; the package public API is declared in __init__.py.
__all__ = ["_cached", "_digest_of", "_table"]


def _cached(fn: Callable[[_T], _R]) -> Callable[[_T], _R]:
    """Cache a property on its immutable node, without retaining nodes globally."""

    @wraps(fn)
    def read(node: _T) -> _R:
        values = vars(node)
        if fn.__name__ not in values:
            values[fn.__name__] = fn(node)
        return cast("_R", values[fn.__name__])

    return read


def _encode(value: _Construction) -> str:
    from .nodes.base import Base

    if isinstance(value, tuple):
        return "(" + ",".join(map(_encode, value)) + ")"
    return (
        f"#{value.digest:032x}" if isinstance(value, Base) else f"{type(value).__name__}:{value!r}"
    )


def _digest_of(identifier: str, values: tuple[_Construction, ...]) -> int:
    data = _encode((identifier, values)).encode()
    return int.from_bytes(hashlib.blake2b(data, digest_size=16).digest(), "big")
