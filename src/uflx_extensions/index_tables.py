# Copyright (C) 2026 Jack S. Hale
#
# This file is part of uflx-core-experimental (https://fenics.github.io/uflx/language/)
#
# SPDX-License-Identifier:    MIT
"""Named tables of indices supplied at evaluation time."""

from typing import final

from uflx.nodes import AbstractExtent, AbstractIndexTable, GatheredIndex
from uflx.tensors import gathered_index
from uflx.typing import Extent, IndexEntry

__all__ = ["IndexArray", "IndexTable", "index_table"]


class IndexArray(AbstractIndexTable):
    """A symbolic input supplied as an integer array of the declared shape."""

    identifier = "uflx_extensions.IndexArray"
    label: str
    _shape: tuple[Extent, ...]

    @property
    def shape(self) -> tuple[Extent, ...]:
        return self._shape

    @property
    def extent(self) -> Extent | None:
        return None

    def __getitem__(self, key: IndexEntry | tuple[IndexEntry, ...]) -> GatheredIndex:
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
    _extent: Extent

    @property
    def extent(self) -> Extent:
        return self._extent


def index_table(label: str, shape: tuple[Extent, ...], extent: Extent | None = None) -> IndexArray:
    """Declare a runtime integer array. Supply extent to check target axes at construction."""
    if extent is None:
        return IndexArray(label, shape)
    return IndexTable(label, shape, extent)
