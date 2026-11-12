from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Never

from .base import Base

if TYPE_CHECKING:
    from .meshes import CellMap


class AbstractCellType(Base):
    @property
    @abstractmethod
    def dimension(self) -> int: ...

    @property
    @abstractmethod
    def num_vertices(self) -> int: ...


class AbstractTopology(Base):
    # A fixed local cell configuration, without physical coordinates.
    @property
    @abstractmethod
    def cell_types(self) -> tuple[AbstractCellType, ...]: ...

    @property
    def dimension(self) -> int:
        return self.cell_types[0].dimension

    def __getitem__(self, cell_slot: int) -> CellMap:
        from ..meshes import cell_map

        return cell_map(self, cell_slot)

    def __iter__(self) -> Never:
        raise TypeError(f"{type(self).__name__} is not iterable")
