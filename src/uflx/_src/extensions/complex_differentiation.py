"""Library compositions for complex projections of the real differential."""

from __future__ import annotations

from ..nodes import (
    AbstractExpression,
    Complex,
    DirectionalDerivative,
    Function,
    Integer,
    ScalarType,
)
from ..nodes.arguments import argument_members, map_argument


def complex_derivative_parts(
    f: AbstractExpression,
    x: Function | tuple[Function, ...],
    a: AbstractExpression | tuple[AbstractExpression, ...],
    h: AbstractExpression | tuple[AbstractExpression, ...],
) -> tuple[AbstractExpression, AbstractExpression]:
    """Return the complex-linear and antilinear directional contributions.

    This composition neither binds h nor evaluates derivatives. For a fresh
    complex direction function h, the returned expressions can be bound with
    entries h and conjugate(h), respectively. Their sum is the full real
    differential; each contribution can be complex even when f is real.
    """
    if any(variable.space.scalar_type != ScalarType.complex for variable in argument_members(x)):
        raise ValueError("complex derivative parts require a complex variable")
    i = Complex(Integer(0), Integer(1))
    delta = DirectionalDerivative(f, x, a, h)
    rotated = DirectionalDerivative(f, x, a, map_argument(lambda value: i * value, h))
    return (delta - i * rotated) / Integer(2), (delta + i * rotated) / Integer(2)
