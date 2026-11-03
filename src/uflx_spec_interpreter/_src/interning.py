from __future__ import annotations

import hashlib
import weakref
from collections.abc import Callable
from functools import wraps
from typing import TYPE_CHECKING, TypeVar, cast

if TYPE_CHECKING:
    from .nodes.base import Base
    from .types import Construction


T = TypeVar("T")
R = TypeVar("R")
table: weakref.WeakValueDictionary[tuple[type[Base], int], Base] = weakref.WeakValueDictionary()


def cached(fn: Callable[[T], R]) -> Callable[[T], R]:
    """Cache a property on its immutable node, without retaining nodes globally."""

    @wraps(fn)
    def read(node: T) -> R:
        values = vars(node)
        if fn.__name__ not in values:
            values[fn.__name__] = fn(node)
        return cast("R", values[fn.__name__])

    return read


def encode(value: Construction) -> str:
    from .nodes.base import Base

    if isinstance(value, tuple):
        return "(" + ",".join(map(encode, value)) + ")"
    return (
        f"#{value.digest:032x}" if isinstance(value, Base) else f"{type(value).__name__}:{value!r}"
    )


def digest_of(identifier: str, values: tuple[Construction, ...]) -> int:
    data = encode((identifier, values)).encode()
    return int.from_bytes(hashlib.blake2b(data, digest_size=16).digest(), "big")
