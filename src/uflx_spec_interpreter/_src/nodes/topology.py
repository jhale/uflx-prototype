from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

from .base import Base

if TYPE_CHECKING:
    from .expressions import AbstractExpression
    from .indices import AbstractExtent, AbstractIndexTable
    from .maps import DomainMap
    from .topology import AbstractCellType, AbstractTopology


class AbstractCellType(Base):
    @property
    @abstractmethod
    def dimension(self) -> int: ...

    @property
    @abstractmethod
    def num_vertices(self) -> int: ...


class AbstractTopology(Base):
    # A set of cells of given cell types, without coordinates.
    @property
    @abstractmethod
    def cell_types(self) -> tuple[AbstractCellType, ...]: ...

    @property
    @abstractmethod
    def num_cells(self) -> tuple[int | AbstractExtent, ...]: ...

    @property
    @abstractmethod
    def sides(self) -> tuple[tuple[tuple[str, AbstractTopology, int], ...], ...]: ...

    @abstractmethod
    def side_tables(
        self, i: int, side: str
    ) -> tuple[AbstractIndexTable, DomainMap, AbstractExpression | None]: ...

    @property
    def dimension(self) -> int:
        return self.cell_types[0].dimension
