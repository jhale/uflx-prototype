"""R01-R03. Resolve side maps and reference normals."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, cast

from ..arithmetic import divide, square_root
from ..geometry import inverse_jacobian
from ..meshes import _cell_of, _inclusion
from ..nodes import (
    CellMap,
    DomainMap,
    Normal,
    SideMap,
)
from ..substitution import _substitute
from ..tensors import transpose
from ..topology import cell_index

if TYPE_CHECKING:
    from ..nodes.base import AbstractExpression
    from ..nodes.indices import GatheredIndex, Index


__all__ = ["_normal", "_side_cell", "_side_maps"]


def _side_cell(
    sm: SideMap, m: DomainMap | CellMap | SideMap
) -> tuple[int, int | Index | GatheredIndex] | None:
    # The group and the entry of m as a cell of the parameter domain of the side map: a cell
    # map of the side topology, or a cell of the side domain.
    if isinstance(m, CellMap):
        return (m.cell_group, m.entry) if m.topology == sm.side_topology else None
    return _cell_of(m) if m.ambient_domain == sm.parameter_domain else None


def _side_maps(
    sm: SideMap, i: int, e: int | Index | GatheredIndex
) -> tuple[CellMap, DomainMap, AbstractExpression | None]:
    # B and a of cell e of group i, and the reference normal of the side tables.
    S, s = sm.side_topology, sm.side
    B, a = _inclusion(sm.ambient_domain, S, i, s, e)
    m_hat = S.side_tables(i, s)[2]
    return (
        B,
        a,
        None
        if m_hat is None
        else cast("AbstractExpression", _substitute(m_hat, {cell_index(S, i): e})),
    )


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
