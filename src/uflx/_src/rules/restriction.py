"""R01-R03. Resolve side maps and reference normals."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from ..arithmetic import divide, square_root
from ..geometry import jacobian
from ..nodes import (
    CellMap,
    Conormal,
    DomainMap,
    SideMap,
)
from ..tensors import inverse, transpose

if TYPE_CHECKING:
    from ..nodes.expressions import AbstractExpression


def side_cell(side_map: SideMap, map: DomainMap | CellMap | SideMap) -> int | None:
    """Identify a cell of the exact restriction domain."""
    if isinstance(map, CellMap) and map.parent == side_map.source:
        return map.cell_slot
    return None


def side_maps(side_map: SideMap, slot: int) -> tuple[CellMap, DomainMap]:
    parent_slot, inclusion = side_map.entries[slot]
    return CellMap(side_map.parent, parent_slot), inclusion


def pull_back_conormal(
    f: Conormal,
    cell: int,
    pb: Callable[[AbstractExpression, DomainMap | CellMap | SideMap], AbstractExpression],
) -> AbstractExpression:
    # K^T m_hat / |K^T m_hat|, with orientation supplied independently of the map.
    outer, inner = side_maps(f.side_map, cell)
    reference_normal = f.reference_conormals[cell]
    w = transpose(pb(inverse(jacobian(outer)), inner)) @ reference_normal
    return divide(w, square_root(w @ w))
