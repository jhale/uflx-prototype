from __future__ import annotations

from typing import TYPE_CHECKING

from .nodes import Complex, Float, Integer, Zero

if TYPE_CHECKING:
    from .nodes.base import AbstractExpression, Base
    from .nodes.domains import AbstractDomain
    from .nodes.indices import AbstractExtent, Index
    from .nodes.topology import AbstractTopology


# The free functions read the static type of an expression.
# Shared implementation helpers; the package public API is declared in __init__.py.
__all__ = ["_arity", "_field_terminal", "_free", "_free_indices", "_shape", "_support"]


def _shape(e: AbstractExpression) -> tuple[int | AbstractExtent, ...]:
    return e.value_shape


def _free(e: AbstractExpression) -> dict[Index, int | AbstractExtent]:
    return dict(e.free_indices)


def _arity(e: AbstractExpression) -> int:
    return len(e.argument_spaces)


def _support(e: AbstractExpression) -> AbstractDomain | AbstractTopology | None:
    return e.domain


def _free_indices(*es: AbstractExpression) -> tuple[tuple[Index, int | AbstractExtent], ...]:
    merged: dict[Index, int | AbstractExtent] = {}
    for e in es:
        merged |= _free(e)
    return tuple(merged.items())


def _field_terminal(e: Base) -> bool:
    # Literal terminals have their own evaluation rows; fields may remain unevaluated.
    return e.is_terminal and not isinstance(e, (Integer, Float, Complex, Zero))
