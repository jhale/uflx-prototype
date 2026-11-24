# Copyright (C) 2026 Jack S. Hale
#
# This file is part of uflx-core-experimental (https://fenics.github.io/uflx/language/)
#
# SPDX-License-Identifier:    MIT
"""Internal generated names preserving structural identity across substitution."""

from __future__ import annotations

from ..nodes import Index
from ..typing.nodes import Construction


def derived_index(purpose: str, *dependencies: Construction) -> Index:
    """Identify an index by its purpose and ordered structural dependencies.

    Keep dependency order and tuple nesting: both are part of existing identity.
    Substitution rewrites nodes inside this label, just as in other derived labels.
    """
    return Index((purpose, *dependencies))


def derived_indices(purpose: str, operands: tuple[Construction, ...], n: int) -> tuple[Index, ...]:
    """Identify each axis by the same operands and its position."""
    return tuple(derived_index(purpose, *operands, m) for m in range(n))
