# Copyright (C) 2026 Jack S. Hale
#
# This file is part of uflx-core-experimental (https://fenics.github.io/uflx/language/)
#
# SPDX-License-Identifier:    MIT

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, final

from .base import Base
from .caching import cached

if TYPE_CHECKING:
    from .types import Construction


class AbstractExtent(Base):
    # An extent known only after construction, which the core compares only by equality.
    pass


@final
class Index(Base):
    identifier = "uflx.Index"
    label: str | tuple[Construction, ...]
    labels = ("label",)


class AbstractIndexTable(Base):
    # A table of index values that the solver gives, which is not an expression.
    @property
    @abstractmethod
    def extent(self) -> int | AbstractExtent | None:
        """Declared size of the axis indexed by each entry, if specified."""
        ...

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
    def extent(self) -> int | AbstractExtent | None:
        return self.table.extent

    @property
    @cached
    def free_indices(self) -> tuple[tuple[Index, int | AbstractExtent], ...]:
        from .validation import entries

        return entries(self.table.shape, self.indices)

    def _validate(self) -> None:
        from .validation import entries

        entries(self.table.shape, self.indices)
