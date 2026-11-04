"""R01-R03. Resolve side maps and reference normals."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, cast

from ..algorithms.substitution import _substitute
from ..arithmetic import divide, square_root
from ..geometry import inverse_jacobian
from ..meshes import _inclusion
from ..nodes import (
    CellMap,
    DomainMap,
    Normal,
    SideMap,
)
from ..tensors import transpose
from ..topology import cell_index

if TYPE_CHECKING:
    from ..nodes.base import AbstractExpression
    from ..nodes.indices import GatheredIndex, Index


__all__ = ["_normal", "_side_cell", "_side_maps"]


def _side_cell(
    side_map: SideMap, map: DomainMap | CellMap | SideMap
) -> tuple[int, int | Index | GatheredIndex] | None:
    """Identify a cell of the side topology, including cells of its physical mesh."""
    if isinstance(map, CellMap) and map.topology == side_map.side_topology:
        return map.cell_group, map.entry
    return None


def _side_maps(
    side_map: SideMap, group: int, entry: int | Index | GatheredIndex
) -> tuple[CellMap, DomainMap, AbstractExpression | None]:
    # B and a of cell e of group i, and the reference normal of the side tables.
    topology, side = side_map.side_topology, side_map.side
    outer, inner = _inclusion(side_map.ambient_domain, topology, group, side, entry)
    reference_normal = topology.side_tables(group, side)[2]
    if reference_normal is not None:
        reference_normal = cast(
            "AbstractExpression",
            _substitute(reference_normal, {cell_index(topology, group): entry}),
        )
    return outer, inner, reference_normal


def _normal(
    f: Normal,
    cell: tuple[int, int | Index | GatheredIndex],
    pb: Callable[[AbstractExpression, DomainMap | CellMap | SideMap], AbstractExpression],
) -> AbstractExpression:
    # K^T m_hat / |K^T m_hat|, with the normal supplied by the side table.
    outer, inner, reference_normal = _side_maps(f.side_map, *cell)
    if reference_normal is None:
        raise TypeError("the side table supplies no reference normal")
    w = transpose(pb(inverse_jacobian(outer), inner)) @ reference_normal
    return divide(w, square_root(w @ w))
