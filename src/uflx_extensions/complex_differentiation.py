# Copyright (C) 2026 Jack S. Hale
#
# This file is part of uflx-core-experimental (https://fenics.github.io/uflx/language/)
#
# SPDX-License-Identifier:    MIT
"""Library compositions for complex projections of the real differential."""

from __future__ import annotations

from uflx.algorithms import argument_leaves, map_argument
from uflx.calculus import directional_derivative
from uflx.nodes import AbstractExpression, Complex, Function, Integer, ScalarType
from uflx.typing import Argument

__all__ = ["complex_derivative_parts"]


def complex_derivative_parts(
    f: AbstractExpression,
    x: Argument[Function],
    a: Argument[AbstractExpression],
    h: Argument[AbstractExpression],
) -> tuple[AbstractExpression, AbstractExpression]:
    """Return the complex-linear and antilinear directional contributions.

    This composition neither binds h nor evaluates derivatives. For a fresh
    complex direction function h, the returned expressions can be bound with
    entries h and conjugate(h), respectively. Their sum is the full real
    differential; each contribution can be complex even when f is real.
    """
    variables = argument_leaves(x)
    if any(variable.space.scalar_type != ScalarType.complex for variable in variables):
        raise ValueError("complex derivative parts require a complex variable")
    i = Complex(Integer(0), Integer(1))
    delta = directional_derivative(f, x, a, h)
    rotated_direction = map_argument(lambda value: i * value, h)
    rotated = directional_derivative(f, x, a, rotated_direction)
    return (delta - i * rotated) / Integer(2), (delta + i * rotated) / Integer(2)
