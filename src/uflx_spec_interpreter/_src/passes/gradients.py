from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from ..arithmetic import multiply
from ..calculus import _rotation_of_gradient, _trace_of_gradient
from ..meshes import _is_cell
from ..names import _derived_index, _derived_indices
from ..nodes import (
    Action,
    ComponentTensor,
    Coordinate,
    Curl,
    Divergence,
    DomainMap,
    Gradient,
    Indexed,
    IndexSum,
    PullBack,
    PushForward,
    Zero,
)
from ..static import _field_terminal, _free_indices, _shape, _support
from ..tensors import _identity
from ..traversal import _bottom_up, _memoized
from .actions import apply_actions
from .derivative_table import _derivative_of_node
from .piola import _piola_law

if TYPE_CHECKING:
    from ..nodes.base import AbstractExpression
    from ..nodes.domains import AbstractDomain
    from ..nodes.indices import Index


# Shared implementation helpers; the package public API is declared in __init__.py.
__all__ = ["_remains", "apply_gradients"]


def _rule(
    partial: Callable[[AbstractExpression, AbstractDomain, Index], AbstractExpression],
    f: AbstractExpression,
    P: AbstractDomain,
    j: Index,
) -> AbstractExpression:
    # Entry j of the last axis of Gradient(f, P), with the leaf rows of apply_gradients first.
    match f:
        case Action(A, p, g):
            return Action(partial(A, P, j), p, g)
        case _ if _support(f) is None:
            return Zero(
                _shape(f),
                (*_free_indices(f), (j, P.ambient_coordinate_domain.dimension)),
                f.argument_spaces,
                f.scalar_type,
            )
        case Coordinate() if P.dimension == P.ambient_coordinate_domain.dimension:
            i = _derived_index("uflx.apply_gradients", f, P)
            return ComponentTensor(
                Indexed(_identity(P.ambient_coordinate_domain.dimension), (i, j)), (i,)
            )
        case PullBack(g, DomainMap() as phi) if _chained(f):
            # The chain rule, because a library gives the gradient of a function on a cell only.
            axes = _derived_indices("uflx.apply_gradients", (f, P), len(_shape(f)))
            r = _derived_index("uflx.apply_gradients", f, P, "r")
            outer = Indexed(PullBack(Gradient(g, phi.ambient_domain), phi), (*axes, r))
            inner = Indexed(partial(phi.parameterisation, P, j), (r,))
            entry = IndexSum(multiply(outer, inner), (r,))
            return ComponentTensor(entry, axes) if axes else entry
        case _ if _remains(f, P):
            axes = _derived_indices("uflx.apply_gradients", (f, P), len(_shape(f)))
            entry = Indexed(Gradient(f, P), (*axes, j))
            return ComponentTensor(entry, axes) if axes else entry
        case _:
            return _derivative_of_node(
                f, lambda b: partial(b, P, j), lambda c: _support(c) is None, []
            )


def _gradient(
    f: AbstractExpression,
    P: AbstractDomain,
    partial: Callable[[AbstractExpression, AbstractDomain, Index], AbstractExpression],
) -> AbstractExpression:
    j = _derived_index("uflx.apply_gradients", f, P, "j")
    axes = _derived_indices("uflx.apply_gradients", (f, P), len(_shape(f)))
    df = partial(f, P, j)
    return ComponentTensor(Indexed(df, axes) if axes else df, (*axes, j))


def _chained(f: AbstractExpression) -> bool:
    # A pull-back by a domain map that is not a cell, whose gradient is the chain rule.
    return isinstance(f, PullBack) and isinstance(f.map, DomainMap) and not _is_cell(f.map)


def _remains(f: AbstractExpression, P: AbstractDomain) -> bool:
    # Field and spatial-operator gradients remain, except the coordinate identity
    # and the chain rule.
    identity = isinstance(f, Coordinate) and P.dimension == P.ambient_coordinate_domain.dimension
    leaf = _field_terminal(f) or isinstance(f, PullBack | Gradient | PushForward)
    return (
        _support(f) is not None
        and (leaf or _piola_law(f) is not None)
        and not identity
        and not _chained(f)
    )


def apply_gradients(f: AbstractExpression) -> AbstractExpression:
    # Inner gradients first, then the rule for the expression of each gradient.
    partial = _memoized(_rule)

    def evaluate(e: AbstractExpression) -> AbstractExpression:
        match e:
            case Divergence(g, p) if _piola_law(e) is None:
                return apply_gradients(_trace_of_gradient(g, p))
            case Curl(g, p) if _piola_law(e) is None:
                return apply_gradients(_rotation_of_gradient(g, p))
            case Gradient(g, p) if not _remains(g, p):
                return apply_actions(_gradient(g, p, partial))
            case _:
                return e

    return _bottom_up(f, evaluate)
