from __future__ import annotations

import hashlib
import weakref
from abc import ABCMeta
from collections.abc import Callable
from functools import wraps
from typing import TYPE_CHECKING, TypeVar, cast

if TYPE_CHECKING:
    from .base import Base
    from .types import _Construction


_T = TypeVar("_T")
_R = TypeVar("_R")
_Node = TypeVar("_Node")
_table: weakref.WeakValueDictionary[tuple[type[Base], int], Base] = weakref.WeakValueDictionary()


# Shared implementation helpers; the package public API is declared in __init__.py.
__all__ = ["_Interned", "_cached", "_digest_of"]


class _Interned(ABCMeta):
    """Intern fully initialized, validated nodes; never reinitialize a cache hit."""

    def __call__(cls: type[_Node], *args: _Construction) -> _Node:
        from .validation import _construction_arguments

        node_class = cast("type[Base]", cls)

        if "identifier" not in node_class.__dict__:
            raise TypeError(f"{node_class.__name__} states no identifier")
        if len(args) != len(node_class.argument_names):
            raise TypeError(
                f"{node_class.__name__} takes the arguments {node_class.argument_names}"
            )
        _construction_arguments(args)
        digest = _digest_of(node_class.identifier, args)
        key = (node_class, digest)
        node = _table.get(key)
        if node is None:
            node = cast("Base", ABCMeta.__call__(cast("ABCMeta", cls), *args))
            vars(node).update(construction_arguments=args, digest=digest)
            _table[key] = node
        return cast("_Node", node)


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
    from .base import Base

    if isinstance(value, tuple):
        return "(" + ",".join(map(_encode, value)) + ")"
    return (
        f"#{value.digest:032x}" if isinstance(value, Base) else f"{type(value).__name__}:{value!r}"
    )


def _digest_of(identifier: str, values: tuple[_Construction, ...]) -> int:
    data = _encode((identifier, values)).encode()
    return int.from_bytes(hashlib.blake2b(data, digest_size=16).digest(), "big")
