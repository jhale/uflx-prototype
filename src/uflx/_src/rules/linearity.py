"""Linearity in selected expression operands."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..nodes import (
    Conjugate,
    Curl,
    Divergence,
    Division,
    Gradient,
    ImaginaryPart,
    Integral,
    PullBack,
    PushForward,
    RealPart,
    Sum,
    Tensor,
)

if TYPE_CHECKING:
    from ..nodes.expressions import AbstractExpression


def linear_fields(expression: AbstractExpression) -> tuple[str, ...] | None:
    """L01-L04. Operands through which a linear operation can commute."""
    match expression:
        case Sum():
            return ("first", "second")
        case Tensor():
            return ("entries",)
        case Division():
            return ("numerator",)
        case Gradient() | Divergence() | Curl() | PushForward() | PullBack() | Integral():
            return ("expression",)
        case Conjugate() | RealPart() | ImaginaryPart():
            return ("operand",)
        case _:
            return expression.commuting or None
