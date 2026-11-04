"""The recursive set of permitted node construction arguments."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeAlias

if TYPE_CHECKING:
    from .base import Base


_Construction: TypeAlias = "Base | int | float | str | tuple[_Construction, ...]"
