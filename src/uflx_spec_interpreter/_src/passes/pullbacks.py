from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, cast

from ..arithmetic import divide, multiply, square_root
from ..calculus import rotation_of_gradient, trace_of_gradient
from ..domains import cell_of, inclusion, is_cell
from ..geometry import density, geometric_map, inverse_jacobian, jacobian
from ..names import derived_index
from ..nodes import (
    Action,
    CellMap,
    ComponentTensor,
    Coordinate,
    Curl,
    Divergence,
    DomainMap,
    Form,
    Gradient,
    Indexed,
    IndexSum,
    Normal,
    PullBack,
    PushForward,
    SideMap,
    TransformationLaw,
)
from ..static import complete_indices, field_terminal, shape, support
from ..substitution import bottom_up, everywhere, freshen, map_values, memoized, substitute
from ..tensors import determinant, transpose
from ..topology import cell_index
from .actions import apply_actions
from .piola import piola_law

if TYPE_CHECKING:
    from ..nodes.base import AbstractExpression
    from ..nodes.indices import GatheredIndex, Index


def rule(
    pb: Callable[[AbstractExpression, DomainMap | CellMap | SideMap], AbstractExpression],
    f: AbstractExpression,
    m: DomainMap | CellMap | SideMap,
) -> AbstractExpression:
    # The rule table of expand_pull_backs, for PullBack(f, m).
    match f:
        case Form():
            return pointwise(f, m, pb)
        case Action(form, position, argument):
            return apply_actions(Action(pb(form, m), position, argument))
        case _ if isinstance(m, SideMap) and support(f) is not None:
            return PullBack(f, m)
        case _ if support(f) is None:
            return complete_indices(PullBack(f, m), f)
        case Coordinate() if isinstance(m, (DomainMap, CellMap)):
            return m.parameterisation
        case PullBack(g, SideMap() as sm) if (cell := side_cell(sm, m)) is not None:
            outer, inner, _ = side_maps(sm, *cell)
            return pb(pb(g, outer), inner)
        case PullBack(_, (SideMap() | CellMap())):
            return PullBack(f, m)
        case PullBack(g, DomainMap() as phi) if isinstance(m, (DomainMap, CellMap)):
            return pb(g, composed(phi, m, pb))
        case Gradient(g, _) if is_cell(m):
            return transformed(
                Gradient(pb(g, m), m.parameter_domain), TransformationLaw(("covariant",), 0), m
            )
        case Gradient():
            return PullBack(f, m)
        case Divergence() | Curl() if (law := piola_law(f)) is not None:
            if not is_cell(m) or m.ambient_domain != f.domain:
                return PullBack(f, m)
            # The Piola identity applies only to a mapped field.
            assert isinstance(f.expression, PushForward)
            expand = trace_of_gradient if isinstance(f, Divergence) else rotation_of_gradient
            reference = on_the_reference_cell(f.expression.expression, m, pb)
            return transformed(expand(reference, m.parameter_domain), law, m)
        case Divergence(g, domain):
            return pb(trace_of_gradient(g, domain), m)
        case Curl(g, domain):
            return pb(rotation_of_gradient(g, domain), m)
        case Normal(sm) if (cell := side_cell(sm, m)) is not None:
            return normal(f, cell, pb)
        case PushForward(g, domain, law) if m.ambient_domain == domain and is_cell(m):
            return pushed_forward(g, law, m, pb)
        case _ if field_terminal(f) or isinstance(f, PushForward):
            return PullBack(f, m)
        case _:
            return pointwise(f, m, pb)


def kept(m: DomainMap | CellMap | SideMap) -> bool:
    # a pull-back stays inside another only by a cell, a cell map or a side map.
    return isinstance(m, SideMap | CellMap) or is_cell(m)


def composed(
    phi: DomainMap,
    m: DomainMap | CellMap,
    pb: Callable[[AbstractExpression, DomainMap | CellMap | SideMap], AbstractExpression],
) -> DomainMap:
    # The map phi after m, so that only a pull-back by a cell keeps a pull-back inside it.
    psi = pb(lowered_copy(pb, phi.parameterisation), m)
    return DomainMap(m.parameter_domain, psi, phi.ambient_domain)


def lowered_copy(
    pb: Callable[[AbstractExpression, DomainMap | CellMap | SideMap], AbstractExpression],
    e: AbstractExpression,
) -> AbstractExpression:
    # The parameterisation of a map that a row builds is lowered as the row builds it, because
    # a lowering pass does not enter maps.
    return bottom_up(e, lambda n: pb(n.expression, n.map) if isinstance(n, PullBack) else n)


def pointwise(
    f: AbstractExpression,
    m: DomainMap | CellMap | SideMap,
    pb: Callable[[AbstractExpression, DomainMap | CellMap | SideMap], AbstractExpression],
) -> AbstractExpression:
    # a binding node is made fresh for the free indices of the map, which occur in the map.
    fresh = freshen(f, [m])
    if isinstance(fresh, Form):
        return Form(pb(fresh.expression, m), fresh.arguments)
    return map_values(fresh, lambda x: pb(x, m), fresh.value_fields)


def derived_axis(
    f: AbstractExpression, m: DomainMap | CellMap | SideMap, axis: tuple[str, int]
) -> Index:
    """Identify a pull-back axis using the shared structural naming rule."""
    return derived_index("uflx.expand_pull_backs", f, m, axis)


def pushed_forward(
    g: AbstractExpression,
    law: TransformationLaw,
    C: CellMap,
    pb: Callable[[AbstractExpression, DomainMap | CellMap | SideMap], AbstractExpression],
) -> AbstractExpression:
    # The law applied to the value of g on the cell C.
    return transformed(on_the_reference_cell(g, C, pb), law, C)


def on_the_reference_cell(
    g: AbstractExpression,
    C: CellMap,
    pb: Callable[[AbstractExpression, DomainMap | CellMap | SideMap], AbstractExpression],
) -> AbstractExpression:
    # The value of g on the topology, read on the cell C through its cell map.
    return pb(g, CellMap(C.topology, C.cell_group, C.entry))


def transformed(ghat: AbstractExpression, law: TransformationLaw, C: CellMap) -> AbstractExpression:
    # K^T on each covariant axis and J on each contravariant axis of ghat, divided by det J, or
    # by the density on a cell of lower dimension, to the weight.
    J, K = jacobian(C), inverse_jacobian(C)
    r, p = len(shape(ghat)), len(law.variances)
    lead = tuple(derived_axis(ghat, C, ("lead", n)) for n in range(r - p))
    out = tuple(derived_axis(ghat, C, ("out", n)) for n in range(p))
    summed = tuple(derived_axis(ghat, C, ("sum", n)) for n in range(p))
    value = Indexed(ghat, (*lead, *summed)) if lead or summed else ghat
    for variance, i, s in zip(law.variances, out, summed, strict=True):
        M = Indexed(K, (s, i)) if variance == "covariant" else Indexed(J, (i, s))
        value = multiply(value, M)
    if summed:
        value = IndexSum(value, summed)
    source, target = geometric_map(C)
    k, n = source.dimension, target.dimension
    scale = determinant(jacobian(C)) if k == n else density(C)
    for _ in range(law.weight):
        value = divide(value, scale)
    for _ in range(-law.weight):
        value = multiply(value, scale)
    axes = (*lead, *out)
    return ComponentTensor(value, axes) if axes else value


def side_cell(
    sm: SideMap, m: DomainMap | CellMap | SideMap
) -> tuple[int, int | Index | GatheredIndex] | None:
    # The group and the entry of m as a cell of the parameter domain of the side map: a cell
    # map of the side topology, or a cell of the side domain.
    if isinstance(m, CellMap):
        return (m.cell_group, m.entry) if m.topology == sm.side_topology else None
    return cell_of(m) if m.ambient_domain == sm.parameter_domain else None


def side_maps(
    sm: SideMap, i: int, e: int | Index | GatheredIndex
) -> tuple[CellMap, DomainMap, AbstractExpression | None]:
    # B and a of cell e of group i, and the reference normal of the side tables.
    S, s = sm.side_topology, sm.side
    B, a = inclusion(sm.ambient_domain, S, i, s, e)
    m_hat = S.side_tables(i, s)[2]
    return (
        B,
        a,
        None
        if m_hat is None
        else cast("AbstractExpression", substitute(m_hat, {cell_index(S, i): e})),
    )


def normal(
    f: Normal,
    cell: tuple[int, int | Index | GatheredIndex],
    pb: Callable[[AbstractExpression, DomainMap | CellMap | SideMap], AbstractExpression],
) -> AbstractExpression:
    # K^T m_hat / |K^T m_hat|, with the normal supplied by the side table.
    outer, inner, reference_normal = side_maps(f.side_map, *cell)
    if reference_normal is None:
        raise TypeError("the side table supplies no reference normal")
    w = transpose(pb(inverse_jacobian(outer), inner)) @ reference_normal
    return divide(w, square_root(w @ w))


def expand_pull_backs(f: AbstractExpression) -> AbstractExpression:
    # Inner pull-backs first, then the rule for the expression of each pull-back.
    pb = memoized(rule)

    def expand(e: AbstractExpression) -> AbstractExpression:
        match e:
            case PullBack(g, m):
                return pb(g, m)
            case _:
                return e

    return bottom_up(f, expand)


def remains(f: AbstractExpression, m: DomainMap | CellMap | SideMap) -> bool:
    # The pull-backs that no row of expand_pull_backs rewrites, which a library reads.
    return not isinstance(f, Action | Form) and (
        isinstance(m, SideMap)
        or field_terminal(f)
        or isinstance(f, PushForward)
        or (isinstance(f, PullBack) and kept(f.map))
        or (isinstance(f, Gradient) and not is_cell(m))
        or (piola_law(f) is not None and (not is_cell(m) or m.ambient_domain != f.domain))
    )


def in_n5(e: AbstractExpression) -> bool:
    return everywhere(e, lambda n: not isinstance(n, PullBack) or remains(n.expression, n.map))
