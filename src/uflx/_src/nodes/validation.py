# Copyright (C) 2026 Jack S. Hale
#
# This file is part of uflx-core-experimental (https://fenics.github.io/uflx/language/)
#
# SPDX-License-Identifier:    MIT
"""Structural construction checks, in the order of the language chapters."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..typing import Argument, NumericalScalarType
from .arguments import argument_pairs

if TYPE_CHECKING:
    from .expressions import AbstractExpression
    from .indices import AbstractExtent, GatheredIndex, Index
    from .spaces import AbstractFunctionSpace


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def extent(n: int | AbstractExtent) -> None:
    if type(n) is bool:
        raise TypeError("extent requires an exact integer")
    if isinstance(n, int):
        require(n >= 0, "extent must be nonnegative")


def merge_indices(
    *groups: tuple[tuple[Index, int | AbstractExtent], ...],
) -> tuple[tuple[Index, int | AbstractExtent], ...]:
    out: dict[Index, int | AbstractExtent] = {}
    for group in groups:
        for i, n in group:
            require(i not in out or out[i] == n, "one index has inconsistent extents")
            out[i] = n
    return tuple(out.items())


def entries(
    shape: tuple[int | AbstractExtent, ...], indices: tuple[int | Index | GatheredIndex, ...]
) -> tuple[tuple[Index, int | AbstractExtent], ...]:
    from . import Index

    if len(indices) != len(shape):
        raise IndexError("entries must cover exactly the rank")
    direct: list[tuple[Index, int | AbstractExtent]] = []
    gathered: list[tuple[Index, int | AbstractExtent]] = []
    for i, n in zip(indices, shape, strict=True):
        if type(i) is bool:
            raise TypeError("index requires an exact integer")
        if isinstance(i, int):
            if i < 0 or (isinstance(n, int) and i >= n):
                raise IndexError("integer entry is outside the axis")
        elif isinstance(i, Index):
            direct.append((i, n))
        else:
            if i.extent is not None and i.extent != n:
                raise IndexError("gather extent differs from the axis")
            gathered.extend(i.free_indices)
    return merge_indices(tuple(direct), tuple(gathered))


def product_scalar_type(space: Argument[AbstractFunctionSpace]) -> NumericalScalarType:
    """Return the common scalar type of a space or a nested product of spaces."""
    if not isinstance(space, tuple):
        return space.scalar_type
    require(len(space) >= 2, "a product needs at least two members")
    types: set[NumericalScalarType] = {product_scalar_type(member) for member in space}
    require(len(types) == 1, "a product requires one scalar type")
    return types.pop()


def admissible(
    V: Argument[AbstractFunctionSpace],
    value: Argument[AbstractExpression],
) -> None:
    from . import ScalarType

    for space, member in argument_pairs(V, value):
        require(member.value_shape == space.value_shape, "argument value shape does not match")
        require(member.argument_spaces == space.argument_spaces, "argument spaces do not match")
        require(
            member.scalar_type in (ScalarType.real, space.scalar_type),
            "argument scalar type does not match",
        )
        require(member.domain in (None, space.domain), "argument domain does not match")
