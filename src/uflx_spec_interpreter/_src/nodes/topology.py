from __future__ import annotations

from abc import abstractmethod

from .base import Base


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
    def cell_types(self) -> tuple: ...

    @property
    @abstractmethod
    def num_cells(self) -> tuple: ...

    @property
    @abstractmethod
    def sides(self) -> tuple: ...

    @abstractmethod
    def side_tables(self, i, side): ...

    @property
    def dimension(self):
        return self.cell_types[0].dimension
