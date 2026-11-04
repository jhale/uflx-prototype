from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from ..algorithms.rewriting import complete_indices, rewrite
from ..algorithms.traversal import memoized
from ..calculus import rotation_of_gradient, trace_of_gradient
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
from ..nodes.static import free_indices
from ..rules.coordinates import chain_partial, chained, piola_law
from ..rules.differentiation import coordinate_partial, derivative_of_node
from ..rules.structural import gradient_component, gradient_remains, retain_gradient
from .actions import apply_actions

if TYPE_CHECKING:
    from ..nodes.domains import AbstractDomain
    from ..nodes.expressions import AbstractExpression
    from ..nodes.indices import Index


def _rule(
    partial: Callable[[AbstractExpression, AbstractDomain, Index], AbstractExpression],
    f: AbstractExpression,
    P: AbstractDomain,
    j: Index,
) -> AbstractExpression:
    # Entry j of the last axis of Gradient(f, P), with the leaf rows of apply_gradients first.
    match f:
        case Action(form, position, argument):
            return Action(partial(form, P, j), position, argument)
        case _ if f.domain is None:
            return Zero(
                f.value_shape,
                (*free_indices(f), (j, P.ambient_coordinate_domain.dimension)),
                f.argument_spaces,
                f.scalar_type,
            )
        case Coordinate() if P.dimension == P.ambient_coordinate_domain.dimension:
            return coordinate_partial(f, P, j)
        case PullBack(g, DomainMap() as phi) if chained(f):
            return chain_partial(f, g, phi, P, j, partial)
        case _ if gradient_remains(f, P):
            return gradient_component(f, P, j)
        case _:
            return derivative_of_node(f, lambda b: partial(b, P, j), lambda c: c.domain is None, [])


def apply_gradients(f: AbstractExpression) -> AbstractExpression:
    # Inner gradients first, then the rule for the expression of each gradient.
    def component_rule(
        partial: Callable[[AbstractExpression, AbstractDomain, Index], AbstractExpression],
        e: AbstractExpression,
        P: AbstractDomain,
        j: Index,
    ) -> AbstractExpression:
        return complete_indices(gradient_component(e, P, j), _rule(partial, e, P, j))

    partial = memoized(component_rule)

    def evaluate(e: AbstractExpression) -> AbstractExpression:
        match e:
            case Divergence(g, p) if piola_law(e) is None:
                return trace_of_gradient(g, p)
            case Curl(g, p) if piola_law(e) is None:
                return rotation_of_gradient(g, p)
            case Gradient(Action(form, position, argument), p):
                return Action(Gradient(form, p), position, argument)
            case Gradient(g, p) if not gradient_remains(g, p):
                return retain_gradient(g, p, partial)
            case _:
                return e

    return apply_actions(rewrite(f, evaluate))
