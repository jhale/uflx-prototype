from __future__ import annotations

from typing import TYPE_CHECKING

from .nodes import GatheredIndex, Index

if TYPE_CHECKING:
    from .nodes.indices import AbstractIndexTable
    from .nodes.types import _Construction


def index(label: str | tuple[_Construction, ...]) -> Index:
    return Index(label)


def gathered_index(
    table: AbstractIndexTable, indices: tuple[int | Index | GatheredIndex, ...]
) -> GatheredIndex:
    return GatheredIndex(table, indices)
