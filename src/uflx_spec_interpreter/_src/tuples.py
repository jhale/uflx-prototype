from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar, cast, overload

_T = TypeVar("_T")
_R = TypeVar("_R")

# Shared implementation helpers; the package public API is declared in __init__.py.
__all__ = ["_each", "_members"]


@overload
def _each(fn: Callable[[_T], _R], x: tuple[_T, ...]) -> tuple[_R, ...]: ...
@overload
def _each(fn: Callable[[_T], _R], x: _T) -> _R: ...
def _each(fn: Callable[[_T], _R], x: _T | tuple[_T, ...]) -> _R | tuple[_R, ...]:
    return tuple(map(fn, cast("tuple[_T, ...]", x))) if isinstance(x, tuple) else fn(x)


def _members(x: _T | tuple[_T, ...]) -> tuple[_T, ...]:
    return cast("tuple[_T, ...]", x) if isinstance(x, tuple) else (x,)
