from __future__ import annotations

from typing import TYPE_CHECKING, Literal, TypeGuard, cast

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
    source: Mesh | AbstractTopology,
    *entries: tuple[CellMap, DomainMap],
) -> SideMap:
    if not entries:
        raise ValueError("at least one cell-map entry is required")
    if any(not isinstance(cast("object", cell), CellMap) for cell, _ in entries):
        raise TypeError("side-map entries require CellMap handles")
    parent = entries[0][0].parent
    if any(cell.parent != parent for cell, _ in entries):
        raise ValueError("side-map entries must have a common parent")
    return SideMap(parent, source, tuple((cell.cell_slot, A) for cell, A in entries))


def side_domain(D: Mesh, S: AbstractTopology, *entries: tuple[int, DomainMap]) -> Mesh:
    return image(PullBack(D.coordinate_function, SideMap(D.topology, S, entries)), D.ambient_domain)


def normal(sm: SideMap, *reference_conormals: AbstractExpression) -> Normal:
    return Normal(sm, reference_conormals)
