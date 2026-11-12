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
    # A fixed local cell configuration, without physical coordinates.
    @property
    @abstractmethod
    def cell_types(self) -> tuple[AbstractCellType, ...]: ...

    @property
    def dimension(self) -> int:
        return self.cell_types[0].dimension
