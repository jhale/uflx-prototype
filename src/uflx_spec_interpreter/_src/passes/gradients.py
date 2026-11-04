from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from ..calculus import _rotation_of_gradient, _trace_of_gradient
from ..nodes import (
    Action,
    Coordinate,
    Curl,
    Divergence,
    DomainMap,
    Gradient,
    PullBack,
    Zero,
)
from ..rewriting import _complete_indices, _rewrite
from ..rules.coordinates import _chain_partial, _chained, _piola_law
from ..rules.differentiation import _coordinate_partial, _derivative_of_node
from ..rules.structural import _action_output, _gradient, _gradient_component, _gradient_remains
from ..static import _free_indices, _shape, _support
from ..traversal import _memoized
from .actions import apply_actions

if TYPE_CHECKING:
    from ..nodes.base import AbstractExpression
    from ..nodes.domains import AbstractDomain
    from ..nodes.indices import Index


# Shared implementation helpers; the package public API is declared in __init__.py.
__all__ = ["apply_gradients"]


def _rule(
    partial: Callable[[AbstractExpression, AbstractDomain, Index], AbstractExpression],
    f: AbstractExpression,
    P: AbstractDomain,
    j: Index,
) -> AbstractExpression:
    # Entry j of the last axis of Gradient(f, P), with the leaf rows of apply_gradients first.
    match f:
        case Action():
            return _action_output(f, lambda A: partial(A, P, j))
        case _ if _support(f) is None:
            return Zero(
                _shape(f),
                (*_free_indices(f), (j, P.ambient_coordinate_domain.dimension)),
                f.argument_spaces,
                f.scalar_type,
            )
        case Coordinate() if P.dimension == P.ambient_coordinate_domain.dimension:
            return _coordinate_partial(f, P, j)
        case PullBack(g, DomainMap() as phi) if _chained(f):
            return _chain_partial(f, g, phi, P, j, partial)
        case _ if _gradient_remains(f, P):
            return _gradient_component(f, P, j)
        case _:
            return _derivative_of_node(
                f, lambda b: partial(b, P, j), lambda c: _support(c) is None, []
            )


def apply_gradients(f: AbstractExpression) -> AbstractExpression:
    # Inner gradients first, then the rule for the expression of each gradient.
    def component_rule(
        partial: Callable[[AbstractExpression, AbstractDomain, Index], AbstractExpression],
        e: AbstractExpression,
        P: AbstractDomain,
        j: Index,
    ) -> AbstractExpression:
        return _complete_indices(_gradient_component(e, P, j), _rule(partial, e, P, j))

    partial = _memoized(component_rule)

    def evaluate(e: AbstractExpression) -> AbstractExpression:
        match e:
            case Divergence(g, p) if _piola_law(e) is None:
                return _trace_of_gradient(g, p)
            case Curl(g, p) if _piola_law(e) is None:
                return _rotation_of_gradient(g, p)
            case Gradient(Action() as g, p):
                return _action_output(g, lambda A: Gradient(A, p))
            case Gradient(g, p) if not _gradient_remains(g, p):
                return _gradient(g, p, partial)
            case _:
                return e

    return apply_actions(_rewrite(f, evaluate))
