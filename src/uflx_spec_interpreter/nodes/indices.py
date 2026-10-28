from __future__ import annotations

from abc import abstractmethod
from functools import cached_property
from typing import TYPE_CHECKING, final

from .base import Base

if TYPE_CHECKING:
    pass


@final
class Index(Base):
    identifier = "uflx.Index"
    label: str | tuple
    opaque = ("label",)
    labels = ("label",)


class AbstractIndexValue(Base):
    # A table of index values that the solver gives, which is not an expression.
    @property
    @abstractmethod
    def extent(self) -> int: ...

    @property
    @abstractmethod
    def shape(self) -> tuple: ...


@final
class IndexTable(AbstractIndexValue):
    identifier = "toy.IndexTable"
    # The toy knows the values, which the specification leaves to a library.
    values: tuple
    extent: int
    opaque = ("values", "extent")

    @cached_property
    def shape(self):
        shape, x = [], self.values
        while isinstance(x, tuple):
            shape.append(len(x))
            x = x[0]
        return tuple(shape)


@final
class GatheredIndex(Base):
    identifier = "uflx.GatheredIndex"
    # An index read from a table at the given entries, an int or an Index for each axis.
    table: AbstractIndexValue
    indices: tuple[int | Index | GatheredIndex, ...]

    @property
    def extent(self):
        return self.table.extent

    @cached_property
    def free_indices(self):
        # Each Index entry takes the extent of its axis of the table, and a nested gather its own.
        out = {}
        for m, j in enumerate(self.indices):
            if isinstance(j, Index):
                out[j] = self.table.shape[m]
            elif isinstance(j, GatheredIndex):
                out |= dict(j.free_indices)
        return tuple(out.items())
