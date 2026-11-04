from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from ..algorithms.rewriting import _complete_indices, _rewrite
from ..algorithms.traversal import _everywhere, _memoized
from ..calculus import _rotation_of_gradient, _trace_of_gradient
from ..meshes import _is_cell
from ..nodes import (
    Absolute,
    Action,
    BroadcastTo,
    CellMap,
    ComponentTensor,
    Conjugate,
    Coordinate,
    Curl,
    Determinant,
    Divergence,
    Division,
    DomainMap,
    Form,
    Gradient,
    ImaginaryPart,
    Indexed,
    IndexSum,
    Inverse,
    Logarithm,
    Normal,
    Power,
    Product,
    PullBack,
    PushForward,
    RealPart,
    SideMap,
    SquareRoot,
    Sum,
    Tensor,
    TransformationLaw,
)
from ..nodes.static import _field_terminal
from ..rules.coordinates import _piola_law, _transformed
from ..rules.pullback_algebra import _composed, _pointwise, _pullback_remains
from ..rules.restriction import _normal, _side_cell, _side_maps
from .actions import apply_actions

if TYPE_CHECKING:
    from ..nodes.expressions import AbstractExpression


def _rule(
    pb: Callable[[AbstractExpression, DomainMap | CellMap | SideMap], AbstractExpression],
    f: AbstractExpression,
    m: DomainMap | CellMap | SideMap,
) -> AbstractExpression:
    # The rule table of expand_pull_backs, for PullBack(f, m).
    match f:
        case Form():
            return _pointwise(f, m, pb)
        case Action(form, position, argument):
            mapped_form = pb(form, m)
            return apply_actions(Action(mapped_form, position, argument))
        case _ if isinstance(m, SideMap) and f.domain is not None:
            return PullBack(f, m)
        case _ if f.domain is None:
            return _complete_indices(PullBack(f, m), f)
        case Coordinate() if isinstance(m, (DomainMap, CellMap)):
            return m.parameterisation
        case PullBack(g, SideMap() as sm) if (cell := _side_cell(sm, m)) is not None:
            outer, inner, _ = _side_maps(sm, *cell)
            return pb(pb(g, outer), inner)
        case PullBack(_, (SideMap() | CellMap())):
            return PullBack(f, m)
        case PullBack(g, DomainMap() as phi) if isinstance(m, (DomainMap, CellMap)):
            return pb(g, _composed(phi, m, pb))
        case Gradient(g, _) if _is_cell(m):
            return _transformed(
                Gradient(pb(g, m), m.parameter_domain), TransformationLaw(("covariant",), 0), m
            )
        case Gradient():
            return PullBack(f, m)
        case Divergence() | Curl() if (law := _piola_law(f)) is not None:
            if not _is_cell(m) or m.ambient_domain != f.domain:
                return PullBack(f, m)
            # The Piola identity applies only to a mapped field.
            assert isinstance(f.expression, PushForward)
            expand = _trace_of_gradient if isinstance(f, Divergence) else _rotation_of_gradient
            reference_cell = CellMap(m.topology, m.cell_group, m.entry)
            reference = pb(f.expression.expression, reference_cell)
            return _transformed(expand(reference, m.parameter_domain), law, m)
        case Divergence(g, domain):
            return pb(_trace_of_gradient(g, domain), m)
        case Curl(g, domain):
            return pb(_rotation_of_gradient(g, domain), m)
        case Normal(sm) if (cell := _side_cell(sm, m)) is not None:
            return _normal(f, cell, pb)
        case PushForward(g, domain, law) if m.ambient_domain == domain and _is_cell(m):
            reference_cell = CellMap(m.topology, m.cell_group, m.entry)
            reference_value = pb(g, reference_cell)
            return _transformed(reference_value, law, m)
        case _ if _field_terminal(f) or isinstance(f, PushForward):
            return PullBack(f, m)
        case (
            Sum()
            | Product()
            | Division()
            | Power()
            | Absolute()
            | Conjugate()
            | RealPart()
            | ImaginaryPart()
            | SquareRoot()
            | Logarithm()
            | Inverse()
            | Determinant()
            | Tensor()
            | BroadcastTo()
            | Indexed()
            | ComponentTensor()
            | IndexSum()
        ):
            # PB06. Only these operations preserve pointwise values under pull-back.
            return _pointwise(f, m, pb)
        case _:
            raise TypeError(f"no pull-back rule for {type(f).__name__}")


def expand_pull_backs(f: AbstractExpression) -> AbstractExpression:
    # Inner pull-backs first, then the rule for the expression of each pull-back.
    pb = _memoized(_rule)

    def expand(e: AbstractExpression) -> AbstractExpression:
        match e:
            case PullBack(g, m):
                return pb(g, m)
            case _:
                return e

    return apply_actions(_rewrite(f, expand))


def _pull_backs_expanded(e: AbstractExpression) -> bool:
    return _everywhere(
        e, lambda n: not isinstance(n, PullBack) or _pullback_remains(n.expression, n.map)
    )
