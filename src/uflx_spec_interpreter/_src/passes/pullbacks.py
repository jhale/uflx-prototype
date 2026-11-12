from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from ..algorithms.rewriting import complete_indices, rewrite, zero_like
from ..algorithms.traversal import everywhere, memoized
from ..calculus import rotation_of_gradient, trace_of_gradient
from ..meshes import is_cell
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
from ..nodes.static import field_terminal
from ..rules.coordinates import piola_law, transformed
from ..rules.pullback_algebra import composed, pointwise, pullback_remains
from ..rules.restriction import pull_back_normal, side_cell, side_maps
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
            return pointwise(f, m, pb)
        case Action(form, position, argument):
            mapped_form = pb(form, m)
            return apply_actions(Action(mapped_form, position, argument))
        case _ if isinstance(m, SideMap) and f.domain is not None:
            return PullBack(f, m)
        case _ if f.domain is None:
            return complete_indices(PullBack(f, m), f)
        case Coordinate() if isinstance(m, (DomainMap, CellMap)):
            return m.parameterisation
        case PullBack(g, SideMap() as sm) if (cell := side_cell(sm, m)) is not None:
            outer, inner = side_maps(sm, cell)
            return pb(pb(g, outer), inner)
        case PullBack(_, (SideMap() | CellMap())):
            return PullBack(f, m)
        case PullBack(g, DomainMap() as phi) if isinstance(m, (DomainMap, CellMap)):
            return pb(g, composed(phi, m, pb))
        case Gradient(g, _) if is_cell(m):
            reference = pb(g, m)
            if reference.domain is None:
                # Apply the constant-gradient rule before cell transport introduces K.
                # Cell transport already requires an admissible full-rank Jacobian.
                original = PullBack(f, m)
                return zero_like(original, original.argument_spaces)
            return transformed(
                Gradient(reference, m.reference_cell), TransformationLaw(("covariant",), 0), m
            )
        case Gradient():
            return PullBack(f, m)
        case Divergence() | Curl() if (law := piola_law(f)) is not None:
            if not is_cell(m) or m.parent != f.domain:
                return PullBack(f, m)
            # The Piola identity applies only to a mapped field.
            assert isinstance(f.expression, PushForward)
            expand = trace_of_gradient if isinstance(f, Divergence) else rotation_of_gradient
            reference_cell = CellMap(m.topology, m.cell_slot)
            reference = pb(f.expression.expression, reference_cell)
            return transformed(expand(reference, m.reference_cell), law, m)
        case Divergence(g, domain):
            return pb(trace_of_gradient(g, domain), m)
        case Curl(g, domain):
            return pb(rotation_of_gradient(g, domain), m)
        case Normal(sm) if (cell := side_cell(sm, m)) is not None:
            return pull_back_normal(f, cell, pb)
        case PushForward(g, domain, law) if is_cell(m) and m.parent == domain:
            reference_cell = CellMap(m.topology, m.cell_slot)
            reference_value = pb(g, reference_cell)
            return transformed(reference_value, law, m)
        case _ if field_terminal(f) or isinstance(f, PushForward):
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
            return pointwise(f, m, pb)
        case _:
            raise TypeError(f"no pull-back rule for {type(f).__name__}")


def expand_pull_backs(f: AbstractExpression) -> AbstractExpression:
    # Inner pull-backs first, then the rule for the expression of each pull-back.
    pb = memoized(_rule)

    def expand(e: AbstractExpression) -> AbstractExpression:
        match e:
            case PullBack(g, m):
                return pb(g, m)
            case _:
                return e

    return apply_actions(rewrite(f, expand))


def pull_backs_expanded(e: AbstractExpression) -> bool:
    return everywhere(
        e, lambda n: not isinstance(n, PullBack) or pullback_remains(n.expression, n.map)
    )
