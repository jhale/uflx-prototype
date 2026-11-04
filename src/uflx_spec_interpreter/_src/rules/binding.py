"""SB16-SB17. Apply explicit forms with capture-avoiding substitution."""

from __future__ import annotations

from typing import TypeVar, cast

from ..nodes import AbstractExpression, Conjugate, Form, Function
from ..nodes.forms import _entry_function
from ..rewriting import _promote
from ..substitution import _freshen, _substitute
from ..tuples import _members

_T = TypeVar("_T")


__all__ = ["_bind"]


def _bind(
    form: Form, q: int, g: AbstractExpression | tuple[AbstractExpression, ...]
) -> AbstractExpression:
    rest = _freshen(Form(form.expression, _without(form.arguments, q)), _members(g))
    pairs = zip(_members(form.arguments[q]), _members(g), strict=True)
    body = cast(
        "AbstractExpression",
        _substitute(rest.expression, dict(_binding(entry, value) for entry, value in pairs)),
    )
    return Form(body, rest.arguments) if rest.arguments else body


def _binding(
    entry: Function | Conjugate, value: AbstractExpression
) -> tuple[Function, AbstractExpression]:
    # The value is that of the entry, so v in an entry conjugate(v) takes the conjugate.
    match entry:
        case Conjugate(v):
            return _entry_function(entry), Conjugate(_promote(v, value))
        case _:
            return entry, _promote(entry, value)


def _without(xs: tuple[_T, ...], i: int) -> tuple[_T, ...]:
    return xs[:i] + xs[i + 1 :]
