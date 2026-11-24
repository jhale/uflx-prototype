# Copyright (C) 2026 Jack S. Hale
#
# This file is part of uflx-core-experimental (https://fenics.github.io/uflx/language/)
#
# SPDX-License-Identifier:    MIT
"""Type aliases that can be used while defining UFLx node classes."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, TypeAlias, TypeVar

if TYPE_CHECKING:
    from ..nodes import Base

NumericalScalarType: TypeAlias = Literal["real", "complex"]
ExpressionScalarType: TypeAlias = Literal["boolean"] | NumericalScalarType
Variance: TypeAlias = Literal["covariant", "contravariant"]

_Leaf = TypeVar("_Leaf", bound="Base | int")
Argument: TypeAlias = _Leaf | tuple["Argument[_Leaf]", ...]
