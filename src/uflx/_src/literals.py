# Copyright (C) 2026 Jack S. Hale
#
# This file is part of uflx-core-experimental (https://fenics.github.io/uflx/language/)
#
# SPDX-License-Identifier:    MIT

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from .nodes import Complex, Float, Integer, Zero
from .typing.nodes import ArgumentSpaces

if TYPE_CHECKING:
    from .nodes.expressions import AbstractExpression
    from .nodes.indices import AbstractExtent, Index


def num(value: int | float | complex) -> AbstractExpression:
    # A convenience for the interpreter examples; expression factories never convert numbers.
    if isinstance(value, complex):
        return Complex(Float(value.real), Float(value.imag))
    return Integer(value) if isinstance(value, int) else Float(value)


def zero(
    value_shape: tuple[int | AbstractExtent, ...],
    free_indices: tuple[tuple[Index, int | AbstractExtent], ...],
    argument_spaces: ArgumentSpaces,
    scalar_type: Literal["real", "complex"],
) -> Zero:
    return Zero(value_shape, free_indices, argument_spaces, scalar_type)
