# Copyright (C) 2026 Jack S. Hale
#
# This file is part of uflx-core-experimental (https://fenics.github.io/uflx/language/)
#
# SPDX-License-Identifier:    MIT
"""Named tables of indices supplied at evaluation time."""

from typing import final

from uflx.nodes import AbstractExtent, AbstractIndexTable, GatheredIndex, Index
from uflx.tensors import gathered_index

__all__ = ["IndexArray", "IndexTable", "index_table"]

_Extent = int | AbstractExtent
_Entry = int | Index | GatheredIndex


class IndexArray(AbstractIndexTable):
    """A symbolic input supplied as an integer array of the declared shape."""

    identifier = "uflx_extensions.IndexArray"
    label: str
    _shape: tuple[_Extent, ...]

    @property
    def shape(self) -> tuple[_Extent, ...]:
        return self._shape

    @property
    def extent(self) -> _Extent | None:
        return None

    def __getitem__(self, key: _Entry | tuple[_Entry, ...]) -> GatheredIndex:
        entries = key if isinstance(key, tuple) else (key,)
        return gathered_index(self, entries)

    def _validate(self) -> None:
        for n in (*self.shape, self.extent):
            if n is None or isinstance(n, AbstractExtent):
                continue
            if type(n) is not int:
                raise TypeError("expected an integer or an abstract extent")
            if n < 0:
                raise ValueError("index table extents must be nonnegative")


@final
class IndexTable(IndexArray):
    """An index array whose entries all target axes of the declared extent."""

    identifier = "uflx_extensions.IndexTable"
    _extent: _Extent

    @property
    def extent(self) -> _Extent:
        return self._extent


def index_table(
    label: str, shape: tuple[_Extent, ...], extent: _Extent | None = None
) -> IndexArray:
    """Declare a runtime integer array. Supply extent to check target axes at construction."""
    if extent is None:
        return IndexArray(label, shape)
    return IndexTable(label, shape, extent)
