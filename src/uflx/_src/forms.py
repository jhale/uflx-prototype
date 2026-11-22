from __future__ import annotations

from typing import TYPE_CHECKING

from .nodes import (
    AbstractExpression,
    Apply,
    Conjugate,
    Form,
    Function,
)

if TYPE_CHECKING:
    from .nodes.spaces import AbstractFunctionSpace
    from .nodes.types import Construction


def function(space: AbstractFunctionSpace, label: str | tuple[Construction, ...]) -> Function:
    return Function(space, label)


def form(
    expression: AbstractExpression,
    arguments: tuple[Function | Conjugate | tuple[Function | Conjugate, ...], ...],
) -> Form:
    return Form(expression, arguments)


def apply(
    form: AbstractExpression,
    arguments: tuple[AbstractExpression | tuple[AbstractExpression, ...], ...],
) -> AbstractExpression:
    if not arguments and not form.argument_spaces:
        return form
    return Apply(form, arguments)
