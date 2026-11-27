# Copyright (C) 2026 Jack S. Hale
#
# This file is part of uflx-core-experimental (https://fenics.github.io/uflx/language/)
#
# SPDX-License-Identifier:    MIT

from __future__ import annotations

from typing import TYPE_CHECKING, TypeGuard, cast

from uflx._src.nodes import (
    AbstractExpression,
    AbstractTopology,
    CellMap,
    Conormal,
    Mesh,
    PullBack,
    PushForward,
    SideMap,
    TransformationLaw,
)

from .typing import Variance

if TYPE_CHECKING:
    from .nodes.domains import AbstractCoordinateDomain
    from .nodes.maps import DomainMap


def cell_map(parent: AbstractTopology | Mesh, cell_slot: int) -> CellMap:
    return CellMap(parent, cell_slot)


def create_mesh(
    coordinate_function: AbstractExpression, ambient_domain: AbstractCoordinateDomain
) -> Mesh:
    return Mesh(coordinate_function, ambient_domain)


def transformation_law(variances: tuple[Variance, ...], weight: int) -> TransformationLaw:
    return TransformationLaw(variances, weight)


def push_forward(expression: AbstractExpression, mesh: Mesh, law: TransformationLaw) -> PushForward:
    """Transport expression values from the mesh topology using the supplied law."""
    if not isinstance(expression, AbstractExpression):
        raise TypeError("push_forward requires an expression")
    return PushForward(expression, mesh, law)


def is_cell(m: DomainMap | CellMap | SideMap) -> TypeGuard[CellMap]:
    """Whether a map selects a cell of a physical mesh."""
    return isinstance(m, CellMap) and isinstance(m.parent, Mesh)


def side_map(
    source: Mesh | AbstractTopology,
    *entries: tuple[CellMap, DomainMap],
) -> SideMap:
    if entries == ():
        raise ValueError("at least one cell-map entry is required")
    if any(not isinstance(cast("object", cell), CellMap) for cell, _ in entries):
        raise TypeError("side-map entries require CellMap handles")
    parent = entries[0][0].parent
    if any(cell.parent != parent for cell, _ in entries):
        raise ValueError("side-map entries must have a common parent")
    return SideMap(parent, source, tuple((cell.cell_slot, A) for cell, A in entries))


def side_domain(parent: Mesh, source: AbstractTopology, *entries: tuple[int, DomainMap]) -> Mesh:
    return create_mesh(
        PullBack(parent.coordinate_function, SideMap(parent.topology, source, entries)),
        parent.ambient_domain,
    )


def conormal(side_map: SideMap, *reference_conormals: AbstractExpression) -> Conormal:
    return Conormal(side_map, reference_conormals)
