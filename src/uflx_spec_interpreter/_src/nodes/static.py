from __future__ import annotations

from typing import TYPE_CHECKING

from . import Complex, Float, Integer, Zero

if TYPE_CHECKING:
    from .base import Base
    from .expressions import AbstractExpression
    from .indices import AbstractExtent, Index


# The free functions read the static type of an expression.
# Shared implementation helpers; the package public API is declared in __init__.py.
__all__ = ["_field_terminal", "_free_indices"]


def _free_indices(*es: AbstractExpression) -> tuple[tuple[Index, int | AbstractExtent], ...]:
    merged: dict[Index, int | AbstractExtent] = {}
    for e in es:
        merged |= dict(e.free_indices)
    return tuple(merged.items())


def _field_terminal(e: Base) -> bool:
    # Literal terminals have their own evaluation rows; fields may remain unevaluated.
    return e.is_terminal and not isinstance(e, (Integer, Float, Complex, Zero))
