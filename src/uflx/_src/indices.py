# Copyright (C) 2026 Jack S. Hale
#
# This file is part of uflx-core-experimental (https://fenics.github.io/uflx/language/)
#
# SPDX-License-Identifier:    MIT

from __future__ import annotations

from typing import TYPE_CHECKING

from .nodes import GatheredIndex, Index
from .typing.nodes import Construction

if TYPE_CHECKING:
    from .nodes.indices import AbstractIndexTable


def index(label: str | tuple[Construction, ...]) -> Index:
    return Index(label)


def gathered_index(
    table: AbstractIndexTable, indices: tuple[int | Index | GatheredIndex, ...]
) -> GatheredIndex:
    return GatheredIndex(table, indices)
