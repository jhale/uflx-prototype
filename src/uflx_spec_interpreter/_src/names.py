"""Internal generated names preserving structural identity across substitution."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from .nodes import Base, Index

if TYPE_CHECKING:
    from .types import Construction


def derived_index(purpose: str, *dependencies: Construction) -> Index:
    """Identify an index by its purpose and ordered structural dependencies.

    Keep dependency order and tuple nesting: both are part of existing identity.
    Substitution rewrites nodes inside this label, just as in other derived labels.
    """
    return Index((purpose, *dependencies))


def derived_indices(purpose: str, operands: tuple[Construction, ...], n: int) -> tuple[Index, ...]:
    """Identify each axis by the same operands and its position."""
    return tuple(derived_index(purpose, *operands, m) for m in range(n))


def renamed_label(
    node: Base, name: Base, avoid: Sequence[Construction]
) -> tuple[Construction, ...]:
    """Record a capture-avoidance identity without referring to its binding node.

    The scoped operands, original name and ordered avoidance context distinguish
    this renaming. Substitution must subsequently treat the label as fixed.
    """
    body = tuple(getattr(node, field) for field in node.scope)
    return ("uflx.rename", name, body, tuple(avoid))


def is_renamed_label(label: object) -> bool:
    """Whether a label records a fixed capture-avoidance identity."""
    return isinstance(label, tuple) and label[:1] == ("uflx.rename",)
