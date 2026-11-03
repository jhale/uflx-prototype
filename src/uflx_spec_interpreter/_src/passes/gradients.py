from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from ..arithmetic import multiply
from ..calculus import _rotation_of_gradient, _trace_of_gradient
from ..domains import _is_cell
from ..names import _derived_index, _derived_indices
from ..nodes import (
    Absolute,
    Action,
    Complex,
    ComponentTensor,
    Conjugate,
    Coordinate,
    Curl,
    Determinant,
    Divergence,
    Division,
    DomainMap,
    Float,
    Form,
    Function,
    Gradient,
    ImaginaryPart,
    Indexed,
    IndexSum,
    Integer,
    Integral,
    Inverse,
    Logarithm,
    Mesh,
    Normal,
    Power,
    Product,
    PullBack,
    PushForward,
    RealPart,
    SquareRoot,
    Sum,
    Tensor,
    Zero,
)
from ..static import _field_terminal, _free_indices, _shape, _support
from ..substitution import _bottom_up, _everywhere, _memoized
from ..tensors import _identity
from .actions import _retained, apply_actions
from .derivative_table import _derivative_of_node
from .piola import _piola_law
from .pullbacks import _remains as _pulled_back_remains
from .pullbacks import expand_pull_backs

if TYPE_CHECKING:
    from ..nodes.base import AbstractExpression
    from ..nodes.domains import AbstractDomain
    from ..nodes.indices import Index


# Shared implementation helpers; the package public API is declared in __init__.py.
__all__ = ["_allowed", "_in_n6", "_lower", "apply_gradients"]


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
    leaf = _field_terminal(f) or isinstance(
        f, PullBack | Gradient | PushForward | Divergence | Curl
    )
    return leaf and not identity and not _chained(f)


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


def _allowed(e: AbstractExpression) -> bool:
    # The core normal form, the output grammar N6.
    match e:
        case (
            Integer()
            | Float()
            | Complex()
            | Zero()
            | Sum()
            | Product()
            | Division()
            | Power()
            | Absolute()
            | Conjugate()
            | RealPart()
            | ImaginaryPart()
            | SquareRoot()
            | Logarithm()
            | Tensor()
            | Indexed()
            | ComponentTensor()
            | IndexSum()
            | Inverse()
            | Determinant()
            | Coordinate()
            | Normal()
            | Function()
            | Form()
            | PushForward()
        ):
            return True
        case _ if _field_terminal(e):
            return True
        case Divergence() | Curl():
            return _piola_law(e) is not None
        case Gradient(f, p):
            return _remains(f, p)
        case Integral(_, p):
            return not isinstance(p, Mesh)
        case PullBack(f, m):
            return _pulled_back_remains(f, m)
        case Action(A, _, _):
            return _retained(A)
        case _:
            return False


def _lower(f: AbstractExpression) -> AbstractExpression:
    # apply_gradients and expand_pull_backs until neither rewrites, the stage of the core after
    # expand_integrals. A graph that the passes leave unchanged is the same object. Gradients on
    # the domains of the forms come first, so a gradient of the coordinates is the identity.
    while (g := expand_pull_backs(apply_gradients(f))) is not f:
        f = g
    return f


def _in_n6(e: AbstractExpression) -> bool:
    return _everywhere(e, _allowed)
