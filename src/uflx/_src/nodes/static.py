from __future__ import annotations

from typing import TYPE_CHECKING

from . import Boolean, Complex, Float, Integer, Zero

if TYPE_CHECKING:
    from .base import Base
    from .expressions import AbstractExpression
    from .indices import AbstractExtent, Index


# The free functions read the static type of an expression.


def free_indices(*es: AbstractExpression) -> tuple[tuple[Index, int | AbstractExtent], ...]:
    """Collect free indices and their extents in first-operand occurrence order."""
    merged: dict[Index, int | AbstractExtent] = {}
    for e in es:
        merged |= dict(e.free_indices)
    return tuple(merged.items())


def field_terminal(e: Base) -> bool:
    # Literal terminals have their own evaluation rows; fields may remain unevaluated.
    return e.is_terminal and not isinstance(e, (Boolean, Integer, Float, Complex, Zero))
