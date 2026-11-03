from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar, cast, overload

T = TypeVar("T")
R = TypeVar("R")


@overload
def each(fn: Callable[[T], R], x: tuple[T, ...]) -> tuple[R, ...]: ...
@overload
def each(fn: Callable[[T], R], x: T) -> R: ...
def each(fn: Callable[[T], R], x: T | tuple[T, ...]) -> R | tuple[R, ...]:
    return tuple(map(fn, cast("tuple[T, ...]", x))) if isinstance(x, tuple) else fn(x)


def members(x: T | tuple[T, ...]) -> tuple[T, ...]:
    return cast("tuple[T, ...]", x) if isinstance(x, tuple) else (x,)
