from __future__ import annotations

from typing import TYPE_CHECKING

from .nodes import (
    AbstractExpression,
    Action,
    Conjugate,
    Form,
    Function,
)

if TYPE_CHECKING:
    from .nodes.spaces import AbstractFunctionSpace
    from .nodes.types import Construction


def function(V: AbstractFunctionSpace, label: str | tuple[Construction, ...]) -> Function:
    return Function(V, label)


def form(
    expression: AbstractExpression,
    arguments: tuple[Function | Conjugate | tuple[Function | Conjugate, ...], ...],
) -> Form:
    return Form(expression, arguments)


def action(
    f: AbstractExpression, position: int, g: AbstractExpression | tuple[AbstractExpression, ...]
) -> Action:
    return Action(f, position, g)
