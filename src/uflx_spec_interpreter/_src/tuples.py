"""Handle a single argument or a tuple of argument members."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, TypeVar, overload

if TYPE_CHECKING:
    from .nodes.base import Base

# Members are nodes or integer indices, never tuples themselves.
_T = TypeVar("_T", bound="Base | int")
_R = TypeVar("_R")

__all__ = ["_argument_members", "_map_argument"]


@overload
def _map_argument(fn: Callable[[_T], _R], argument: tuple[_T, ...]) -> tuple[_R, ...]: ...
@overload
def _map_argument(fn: Callable[[_T], _R], argument: _T) -> _R: ...
def _map_argument(fn: Callable[[_T], _R], argument: _T | tuple[_T, ...]) -> _R | tuple[_R, ...]:
    """Apply fn to each member, preserving single or tuple structure."""
    if isinstance(argument, tuple):
        return tuple(fn(member) for member in argument)
    return fn(argument)


def _argument_members(argument: _T | tuple[_T, ...]) -> tuple[_T, ...]:
    """Return a tuple of members, wrapping a single argument."""
    if isinstance(argument, tuple):
        return argument
    return (argument,)
