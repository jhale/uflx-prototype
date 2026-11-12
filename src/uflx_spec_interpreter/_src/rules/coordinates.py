"""Coordinate changes, tensor transformations and Piola identities."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from ..algorithms.names import derived_index, derived_indices
from ..arithmetic import divide, multiply
from ..geometry import density, geometric_map, inverse_jacobian, jacobian
from ..nodes import (
    CellMap,
    Curl,
    Divergence,
    DomainMap,
    Gradient,
    PullBack,
    PushForward,
    SideMap,
    TransformationLaw,
)
from ..tensors import component_tensor, determinant, index_sum, indexed

if TYPE_CHECKING:
    from ..nodes.domains import AbstractDomain
    from ..nodes.expressions import AbstractExpression
    from ..nodes.indices import Index


def piola_law(expression: AbstractExpression) -> TransformationLaw | None:
    """Return the output transformation law, or None when no identity applies.

    This inspects the differential and mapped field only. A pull-back must
    separately establish that its map selects a cell of the matching mesh.
    """
    match expression:
        case Divergence(PushForward(_, d, TransformationLaw(("contravariant",), 1)), p) if p == d:
            return TransformationLaw((), 1)
        case Curl(PushForward(_, d, TransformationLaw(("covariant",), 0)), p) if (
            p == d and d.dimension == d.ambient_coordinate_domain.dimension
        ):
            return TransformationLaw(("contravariant",) * len(expression.value_shape), 1)
        case _:
            return None


def chained(f: AbstractExpression) -> bool:
    # A pull-back by a domain map, whose gradient is the chain rule.
    return isinstance(f, PullBack) and isinstance(f.map, DomainMap)


def chain_partial(
    f: AbstractExpression,
    g: AbstractExpression,
    phi: DomainMap,
    P: AbstractDomain,
    j: Index,
    partial: Callable[[AbstractExpression, AbstractDomain, Index], AbstractExpression],
) -> AbstractExpression:
    """CC01. Transport a component derivative through a domain map."""
    axes = derived_indices("uflx.apply_gradients", (f, P), len(f.value_shape))
    r = derived_index("uflx.apply_gradients", f, P, "r")
    outer = indexed(PullBack(Gradient(g, phi.ambient_domain), phi), (*axes, r))
    inner = indexed(partial(phi.parameterisation, P, j), (r,))
    return component_tensor(index_sum(outer * inner, (r,)), axes)


def _derived_axis(
    f: AbstractExpression, m: DomainMap | CellMap | SideMap, axis: tuple[str, int]
) -> Index:
    """Identify a pull-back axis using the shared structural naming rule."""
    return derived_index("uflx.expand_pull_backs", f, m, axis)


def transformed(ghat: AbstractExpression, law: TransformationLaw, C: CellMap) -> AbstractExpression:
    # K^T on each covariant axis and J on each contravariant axis of ghat, divided by det J, or
    # by the density on a cell of lower dimension, to the weight.
    J, K = jacobian(C), inverse_jacobian(C)
    rank = len(ghat.value_shape)
    mapped_axes = len(law.variances)
    lead = tuple(_derived_axis(ghat, C, ("lead", n)) for n in range(rank - mapped_axes))
    out = tuple(_derived_axis(ghat, C, ("out", n)) for n in range(mapped_axes))
    summed = tuple(_derived_axis(ghat, C, ("sum", n)) for n in range(mapped_axes))
    value = indexed(ghat, (*lead, *summed))
    for variance, output_axis, summed_axis in zip(law.variances, out, summed, strict=True):
        if variance == "covariant":
            entry = indexed(K, (summed_axis, output_axis))
        else:
            entry = indexed(J, (output_axis, summed_axis))
        value = multiply(value, entry)
    value = index_sum(value, summed)
    source, target = geometric_map(C)
    k, n = source.dimension, target.dimension
    scale = determinant(jacobian(C)) if k == n else density(C)
    for _ in range(law.weight):
        value = divide(value, scale)
    for _ in range(-law.weight):
        value = multiply(value, scale)
    axes = (*lead, *out)
    return component_tensor(value, axes)
