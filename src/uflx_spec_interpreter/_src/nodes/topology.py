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
