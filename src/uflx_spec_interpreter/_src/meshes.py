from __future__ import annotations

from typing import TYPE_CHECKING, Literal, TypeGuard

from uflx_spec_interpreter._src.nodes import (
    AbstractTopology,
    CellMap,
    Mesh,
    Normal,
    PullBack,
    PushForward,
    SideMap,
    TransformationLaw,
)

if TYPE_CHECKING:
    from uflx_spec_interpreter._src.nodes.expressions import AbstractExpression

    from .nodes.domains import AbstractCoordinateDomain
    from .nodes.maps import DomainMap


def cell_map(parent: AbstractTopology | Mesh, cell_slot: int) -> CellMap:
    return CellMap(parent, cell_slot)


def image(x: AbstractExpression, Q: AbstractCoordinateDomain) -> Mesh:
    return Mesh(x, Q)


def transformation_law(
    variances: tuple[Literal["covariant", "contravariant"], ...], weight: int
) -> TransformationLaw:
    return TransformationLaw(variances, weight)


def push_forward(f: AbstractExpression, D: Mesh, m: TransformationLaw) -> PushForward:
    return PushForward(f, D, m)


def is_cell(m: DomainMap | CellMap | SideMap) -> TypeGuard[CellMap]:
    """Whether a map selects a cell of a physical mesh."""
    return isinstance(m, CellMap) and isinstance(m.parent, Mesh)


def side_map(
    parent: Mesh | AbstractTopology,
    source: Mesh | AbstractTopology,
    *entries: tuple[int, DomainMap],
) -> SideMap:
    return SideMap(parent, source, entries)


def side_domain(D: Mesh, S: AbstractTopology, *entries: tuple[int, DomainMap]) -> Mesh:
    return image(
        PullBack(D.coordinate_function, side_map(D.topology, S, *entries)), D.ambient_domain
    )


def normal(sm: SideMap, *reference_conormals: AbstractExpression) -> Normal:
    return Normal(sm, reference_conormals)


def compose(outer: SideMap, inner: SideMap) -> SideMap:
    """Compose evaluation relations, inner first, without transporting orientation."""
    from .maps import domain_map

    if inner.parent != outer.source:
        raise ValueError("side-map composition requires matching intermediate endpoints")
    entries: list[tuple[int, DomainMap]] = []
    for j, A in inner.entries:
        k, B = outer.entries[j]
        entries.append(
            (k, domain_map(A.parameter_domain, PullBack(B.parameterisation, A), B.ambient_domain))
        )
    return SideMap(outer.parent, inner.source, tuple(entries))
