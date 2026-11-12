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
    parent: Mesh | AbstractTopology, restriction_domain: Mesh | AbstractTopology, side: str
) -> SideMap:
    return SideMap(parent, restriction_domain, side)


def side_domain(D: Mesh, S: AbstractTopology, s: str) -> Mesh:
    x = D.coordinate_function
    return image(PullBack(x, side_map(D.topology, S, s)), D.ambient_domain)


def normal(sm: SideMap) -> Normal:
    return Normal(sm)
