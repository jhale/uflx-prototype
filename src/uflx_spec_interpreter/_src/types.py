"""The recursive set of permitted node construction arguments."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeAlias

if TYPE_CHECKING:
    from .nodes.base import Base


_Construction: TypeAlias = "Base | int | float | str | tuple[_Construction, ...]"
# Shared implementation helpers; the package public API is declared in __init__.py.
__all__ = ["_Construction"]
