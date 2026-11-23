"""Named tables of indices supplied at evaluation time."""

from typing import final

from uflx.nodes import AbstractExtent, AbstractIndexTable, GatheredIndex, Index
from uflx.tensors import gathered_index

__all__ = ["IndexTable", "index_table"]

_Extent = int | AbstractExtent
_Entry = int | Index | GatheredIndex


@final
class IndexTable(AbstractIndexTable):
    """A symbolic table whose entries are indices in [0, extent)."""

    identifier = "uflx_extensions.IndexTable"
    label: str
    _shape: tuple[_Extent, ...]
    _extent: _Extent

    @property
    def shape(self) -> tuple[_Extent, ...]:
        return self._shape

    @property
    def extent(self) -> _Extent:
        return self._extent

    def __getitem__(self, key: _Entry | tuple[_Entry, ...]) -> GatheredIndex:
        entries = key if isinstance(key, tuple) else (key,)
        return gathered_index(self, entries)

    def _validate(self) -> None:
        for n in (*self.shape, self.extent):
            if isinstance(n, AbstractExtent):
                continue
            if type(n) is not int:
                raise TypeError("expected an integer or an abstract extent")
            if n < 0:
                raise ValueError("index table extents must be nonnegative")


def index_table(label: str, shape: tuple[_Extent, ...], extent: _Extent) -> IndexTable:
    """Declare an integer table of the given shape and exclusive upper bound."""
    return IndexTable(label, shape, extent)
