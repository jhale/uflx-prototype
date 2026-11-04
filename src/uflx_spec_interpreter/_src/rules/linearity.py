"""Linearity in selected expression operands."""

from __future__ import annotations

from collections.abc import Callable, Sequence
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
from .structural import _commute

if TYPE_CHECKING:
    from ..nodes.base import AbstractExpression
    from ..types import _Construction


__all__ = ["_action_linearity", "_linear"]


def _linear(
    f: AbstractExpression,
    transform: Callable[[AbstractExpression], AbstractExpression],
    independent: Callable[[AbstractExpression], bool],
    avoid: Sequence[_Construction],
    at: Callable[[AbstractExpression], AbstractExpression] = lambda e: e,
) -> AbstractExpression | None:
    """L01-L04. Unchanged value operands must be independent."""
    match f:
        case Sum():
            fields = ("first", "second")
        case Tensor():
            fields = ("entries",)
        case Division(_, denominator) if independent(denominator):
            fields = ("first",)
        case Gradient() | Divergence() | Curl() | PushForward() | PullBack() | Integral():
            fields = ("expression",)
        case Conjugate() | RealPart() | ImaginaryPart():
            fields = ("operand",)
        case _ if f.commuting:
            # Extensions may state additional linear operands.
            fields = f.commuting
        case _:
            return None
    from ..nodes import AbstractExpression
    from ..traversal import _nodes_in

    unchanged = tuple(name for name in f.value_fields if name not in fields)
    if any(
        isinstance(x, AbstractExpression) and not independent(x) for x in _nodes_in(f, unchanged)
    ):
        return None
    return _commute(f, transform, fields, avoid, at)


def _action_linearity(
    f: AbstractExpression,
    p: int,
    g: AbstractExpression | tuple[AbstractExpression, ...],
    act: Callable[
        [AbstractExpression, int, AbstractExpression | tuple[AbstractExpression, ...]],
        AbstractExpression,
    ],
) -> AbstractExpression | None:
    """L05, L07, then L01-L03 for valid action slots (SH17)."""
    from ..nodes import Zero
    from ..static import _free_indices
    from ..tuples import _each, _members

    match f:
        case Conjugate(e):
            return Conjugate(act(e, p, _each(Conjugate, g)))
        case Zero(s, _, args):
            return Zero(s, _free_indices(f, *_members(g)), args[:p] + args[p + 1 :], f.scalar_type)
        case _:
            return _linear(f, lambda b: act(b, p, g), lambda b: not b.argument_spaces, _members(g))
