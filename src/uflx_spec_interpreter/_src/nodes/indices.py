from __future__ import annotations

from abc import abstractmethod
from functools import cached_property
from typing import final

from .base import Base


class AbstractExtent(Base):
    # An extent known only after construction, which the core compares only by equality.
    pass


@final
class Index(Base):
    identifier = "uflx.Index"
    label: str | tuple
    opaque = ("label",)
    labels = ("label",)


class AbstractIndexTable(Base):
    # A table of index values that the solver gives, which is not an expression.
    @property
    @abstractmethod
    def extent(self) -> int: ...

    @property
    @abstractmethod
    def shape(self) -> tuple: ...


@final
class GatheredIndex(Base):
    identifier = "uflx.GatheredIndex"
    # An index read from a table at the given entries, an int or an Index for each axis.
    table: AbstractIndexTable
    indices: tuple[int | Index | GatheredIndex, ...]

    @property
    def extent(self):
        return self.table.extent

    @cached_property
    def free_indices(self):
        from ..validation import entries

        return entries(self.table.shape, self.indices)
