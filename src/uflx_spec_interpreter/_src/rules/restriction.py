"""R01-R03. Resolve side maps and reference normals."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from ..arithmetic import divide, square_root
from ..geometry import inverse_jacobian
from ..nodes import (
    CellMap,
    DomainMap,
    Normal,
    SideMap,
)
from ..tensors import transpose

if TYPE_CHECKING:
    from ..nodes.expressions import AbstractExpression


def side_cell(side_map: SideMap, map: DomainMap | CellMap | SideMap) -> int | None:
    """Identify a cell of the exact restriction domain."""
    if isinstance(map, CellMap) and map.parent == side_map.restriction_domain:
        return map.cell_slot
    return None


def side_maps(
    side_map: SideMap, slot: int
) -> tuple[CellMap, DomainMap, AbstractExpression | None]:
    topology, side = side_map.side_topology, side_map.side
    _, _, parent_slot = next(s for s in topology.sides[slot] if s[0] == side)
    inner, reference_normal = topology.side_data(slot, side)
    return CellMap(side_map.parent, parent_slot), inner, reference_normal


def pull_back_normal(
    f: Normal,
    cell: int,
    pb: Callable[[AbstractExpression, DomainMap | CellMap | SideMap], AbstractExpression],
) -> AbstractExpression:
    # K^T m_hat / |K^T m_hat|, with the normal supplied by the side table.
    outer, inner, reference_normal = side_maps(f.side_map, cell)
    if reference_normal is None:
        raise TypeError("the side table supplies no reference normal")
    w = transpose(pb(inverse_jacobian(outer), inner)) @ reference_normal
    return divide(w, square_root(w @ w))
