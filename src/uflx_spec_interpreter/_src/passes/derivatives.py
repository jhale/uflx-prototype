from __future__ import annotations

from collections.abc import Callable
from functools import cache
from typing import TYPE_CHECKING

from ..nodes import DirectionalDerivative
from ..static import complete_indices, promote, zero_like
from ..substitution import bottom_up, everywhere, memoized, occurs, substitute_values
from ..tuples import members
from .actions import apply_actions
from .derivative_table import derivative_of_node

if TYPE_CHECKING:
    from ..nodes.base import AbstractExpression
    from ..nodes.forms import Function


def deriv(
    f: AbstractExpression,
    x: Function | tuple[Function, ...],
    a: AbstractExpression | tuple[AbstractExpression, ...],
    h: AbstractExpression | tuple[AbstractExpression, ...],
) -> AbstractExpression:
    # Evaluate undifferentiated operands at the point; inserted directions stay unchanged.
    xs, points, directions = members(x), members(a), members(h)
    points = tuple(promote(v, p) for v, p in zip(xs, points, strict=True))
    directions = tuple(promote(v, w) for v, w in zip(xs, directions, strict=True))
    mapping = dict(zip(xs, points, strict=True))

    @cache
    def at(e: AbstractExpression) -> AbstractExpression:
        return substitute_values(e, mapping)

    def constant(e: AbstractExpression) -> bool:
        return not any(occurs(v, e) for v in xs)

    def differentiate(
        d: Callable[[AbstractExpression], AbstractExpression], e: AbstractExpression
    ) -> AbstractExpression:
        for variable, direction in zip(xs, directions, strict=True):
            if e is variable:
                return direction
        if constant(e):
            return zero_like(e, e.argument_spaces, *points, *directions)
        return derivative_of_node(e, d, constant, [*xs, *points, *directions], at)

    return complete_indices(DirectionalDerivative(f, x, a, h), memoized(differentiate)(f))


def apply_derivatives(f: AbstractExpression) -> AbstractExpression:
    # Derivative evaluation does no optional algebraic collection.
    differential = cache(deriv)

    def evaluate(e: AbstractExpression) -> AbstractExpression:
        match e:
            case DirectionalDerivative(g, x, a, h):
                return differential(g, x, a, h)
            case _:
                return e

    # Applying a newly exposed action can reveal another derivative.
    out = apply_actions(bottom_up(f, evaluate))
    return out if in_n3(out) else apply_derivatives(out)


def in_n3(e: AbstractExpression) -> bool:
    return everywhere(e, lambda n: not isinstance(n, DirectionalDerivative))
