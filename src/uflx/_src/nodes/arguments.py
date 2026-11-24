# Copyright (C) 2026 Jack S. Hale
#
# This file is part of uflx-core-experimental (https://fenics.github.io/uflx/language/)
#
# SPDX-License-Identifier:    MIT
"""Handle a single argument or a tuple of argument members."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING, TypeAlias, TypeVar

if TYPE_CHECKING:
    from .arithmetic import Conjugate
    from .base import Base
    from .forms import Function
    from .indices import Index
    from .spaces import AbstractFunctionSpace

_T = TypeVar("_T", bound="Base | int")
_U = TypeVar("_U", bound="Base | int")
_R = TypeVar("_R", bound="Base | int")

Argument: TypeAlias = _T | tuple["Argument[_T]", ...]
ArgumentSpaces: TypeAlias = "tuple[Argument[AbstractFunctionSpace], ...]"
FormEntry: TypeAlias = "Argument[Function | Conjugate]"
BindingEntry: TypeAlias = "FormEntry | Index"


def map_argument(fn: Callable[[_T], _R], argument: Argument[_T]) -> Argument[_R]:
    """Apply fn to the leaves of one argument, preserving its product structure."""
    if isinstance(argument, tuple):
        return tuple(map_argument(fn, member) for member in argument)
    return fn(argument)


def argument_members(argument: _T | tuple[_T, ...]) -> tuple[_T, ...]:
    """Return the immediate members, wrapping a single value."""
    return argument if isinstance(argument, tuple) else (argument,)


def argument_leaves(argument: Argument[_T]) -> tuple[_T, ...]:
    """Return the leaves of one argument in declaration order."""
    if isinstance(argument, tuple):
        return tuple(leaf for member in argument for leaf in argument_leaves(member))
    return (argument,)


def argument_pairs(left: Argument[_T], right: Argument[_U]) -> Iterator[tuple[_T, _U]]:
    """Pair leaves at the same position in two identically grouped arguments."""
    if isinstance(left, tuple) and len(left) < 2:
        raise ValueError("a product needs at least two members")
    if isinstance(left, tuple) and isinstance(right, tuple) and len(left) == len(right):
        for a, b in zip(left, right, strict=True):
            yield from argument_pairs(a, b)
    elif isinstance(left, tuple) or isinstance(right, tuple):
        raise ValueError("argument grouping differs: tuple structure does not match")
    else:
        yield left, right


def binding_name(member: Function | Conjugate | Index) -> Function | Index:
    from .forms import entry_function
    from .indices import Index

    return member if isinstance(member, Index) else entry_function(member)
