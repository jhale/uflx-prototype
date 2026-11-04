from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, final

from .base import Base
from .interning import _cached

if TYPE_CHECKING:
    from .types import _Construction


class AbstractExtent(Base):
    # An extent known only after construction, which the core compares only by equality.
    pass


@final
class Index(Base):
    identifier = "uflx.Index"
    label: str | tuple[_Construction, ...]
    labels = ("label",)


class AbstractIndexTable(Base):
    # A table of index values that the solver gives, which is not an expression.
    @property
    @abstractmethod
    def extent(self) -> int | AbstractExtent: ...

    @property
    @abstractmethod
    def shape(self) -> tuple[int | AbstractExtent, ...]: ...


@final
class GatheredIndex(Base):
    index_fields = ("indices",)
    identifier = "uflx.GatheredIndex"
    # An index read from a table at the given entries, an int or an Index for each axis.
    table: AbstractIndexTable
    indices: tuple[int | Index | GatheredIndex, ...]

    @property
    def extent(self) -> int | AbstractExtent:
        return self.table.extent

    @property
    @_cached
    def free_indices(self) -> tuple[tuple[Index, int | AbstractExtent], ...]:
        from .validation import _entries

        return _entries(self.table.shape, self.indices)

    def _validate(self) -> None:
        from .validation import _entries

        _entries(self.table.shape, self.indices)
