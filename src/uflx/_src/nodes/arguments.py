# Copyright (C) 2026 Jack S. Hale
#
# This file is part of uflx-core-experimental (https://fenics.github.io/uflx/language/)
#
# SPDX-License-Identifier:    MIT
"""Handle a single argument or a tuple of argument members."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, TypeAlias, TypeVar, overload

if TYPE_CHECKING:
    from .arithmetic import Conjugate
    from .base import Base
    from .forms import Function
    from .indices import Index

FormEntry: TypeAlias = "Function | Conjugate | tuple[Function | Conjugate, ...]"
BindingEntry: TypeAlias = "FormEntry | Index"

# Members are nodes or integer indices, never tuples themselves.
_T = TypeVar("_T", bound="Base | int")
_R = TypeVar("_R")


@overload
def map_argument(fn: Callable[[_T], _R], argument: tuple[_T, ...]) -> tuple[_R, ...]: ...
@overload
def map_argument(fn: Callable[[_T], _R], argument: _T) -> _R: ...
def map_argument(fn: Callable[[_T], _R], argument: _T | tuple[_T, ...]) -> _R | tuple[_R, ...]:
    """Apply fn to each member, preserving single or tuple structure."""
    if isinstance(argument, tuple):
        return tuple(fn(member) for member in argument)
    return fn(argument)


def argument_members(argument: _T | tuple[_T, ...]) -> tuple[_T, ...]:
    """Return a tuple of members, wrapping a single argument."""
    if isinstance(argument, tuple):
        return argument
    return (argument,)


def binding_name(member: Function | Conjugate | Index) -> Function | Index:
    from .forms import entry_function
    from .indices import Index

    return member if isinstance(member, Index) else entry_function(member)
