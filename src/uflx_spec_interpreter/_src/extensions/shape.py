"""Compatibility conveniences; shape differentiation uses the core derivative."""

from __future__ import annotations

from ..nodes import AbstractExpression, DirectionalDerivative, Function
from ..passes.lowering import lower


def shape_derivative(
    f: AbstractExpression,
    x: Function | tuple[Function, ...],
    h: AbstractExpression | tuple[AbstractExpression, ...],
) -> DirectionalDerivative:
    """Construct the ordinary derivative in the geometry direction h."""
    return DirectionalDerivative(f, x, x, h)


def lower_with_shape_derivatives(f: AbstractExpression) -> AbstractExpression:
    """Use standard core lowering; retained for existing extension callers."""
    return lower(f)
