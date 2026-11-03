from __future__ import annotations

from typing import TYPE_CHECKING

from .names import derived_index

if TYPE_CHECKING:
    from .nodes.indices import Index
    from .nodes.topology import AbstractTopology


def cell_index(T: AbstractTopology, j: int) -> Index:

    if type(j) is not int:
        raise TypeError("cell_index takes a topology and an integer")
    if not 0 <= j < len(T.cell_types):
        raise ValueError("invalid cell group")
    return derived_index("uflx.cell_index", T, j, 0)
