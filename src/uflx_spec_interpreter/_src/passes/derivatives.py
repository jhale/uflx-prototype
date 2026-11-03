from __future__ import annotations

from collections.abc import Callable
from functools import cache
from typing import TYPE_CHECKING

from ..nodes import DirectionalDerivative
from ..rewriting import _complete_indices, _promote, _zero_like
from ..substitution import _occurs, _substitute_values
from ..traversal import _bottom_up, _everywhere, _memoized
from ..tuples import _members
from .actions import apply_actions
from .derivative_table import _derivative_of_node

if TYPE_CHECKING:
    from ..nodes.base import AbstractExpression
    from ..nodes.forms import Function


# Shared implementation helpers; the package public API is declared in __init__.py.
__all__ = ["_deriv", "_derivatives_complete", "apply_derivatives"]


def _deriv(
    f: AbstractExpression,
    x: Function | tuple[Function, ...],
    a: AbstractExpression | tuple[AbstractExpression, ...],
    h: AbstractExpression | tuple[AbstractExpression, ...],
) -> AbstractExpression:
    # Evaluate undifferentiated operands at the point; inserted directions stay unchanged.
    xs, points, directions = _members(x), _members(a), _members(h)
    points = tuple(_promote(v, p) for v, p in zip(xs, points, strict=True))
    directions = tuple(_promote(v, w) for v, w in zip(xs, directions, strict=True))
    mapping = dict(zip(xs, points, strict=True))

    @cache
    def at(e: AbstractExpression) -> AbstractExpression:
        return _substitute_values(e, mapping)

    def constant(e: AbstractExpression) -> bool:
        return not any(_occurs(v, e) for v in xs)

    def differentiate(
        d: Callable[[AbstractExpression], AbstractExpression], e: AbstractExpression
    ) -> AbstractExpression:
        for variable, direction in zip(xs, directions, strict=True):
            if e is variable:
                return direction
        if constant(e):
            return _zero_like(e, e.argument_spaces, *points, *directions)
        return _derivative_of_node(e, d, constant, [*xs, *points, *directions], at)

    return _complete_indices(DirectionalDerivative(f, x, a, h), _memoized(differentiate)(f))


def apply_derivatives(f: AbstractExpression) -> AbstractExpression:
    # Derivative evaluation does no optional algebraic collection.
    differential = cache(_deriv)

    def evaluate(e: AbstractExpression) -> AbstractExpression:
        match e:
            case DirectionalDerivative(g, x, a, h):
                return differential(g, x, a, h)
            case _:
                return e

    # Applying a newly exposed action can reveal another derivative.
    out = apply_actions(_bottom_up(f, evaluate))
    return out if _derivatives_complete(out) else apply_derivatives(out)


def _derivatives_complete(e: AbstractExpression) -> bool:
    return _everywhere(e, lambda n: not isinstance(n, DirectionalDerivative))
