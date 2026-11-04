"""SB16-SB17. Apply explicit forms with capture-avoiding substitution."""

from __future__ import annotations

from typing import cast

from ..nodes import AbstractExpression, Conjugate, Form, Function
from ..nodes.forms import _entry_function
from ..rewriting import _promote
from ..substitution import _freshen, _substitute
from ..tuples import _members

__all__ = ["_bind"]


def _bind(
    form: Form, position: int, argument: AbstractExpression | tuple[AbstractExpression, ...]
) -> AbstractExpression:
    remaining_arguments = form.arguments[:position] + form.arguments[position + 1 :]
    remaining_form = _freshen(Form(form.expression, remaining_arguments), _members(argument))
    pairs = zip(_members(form.arguments[position]), _members(argument), strict=True)
    replacements = dict(_binding(entry, value) for entry, value in pairs)
    replaced_body = cast("AbstractExpression", _substitute(remaining_form.expression, replacements))
    return (
        Form(replaced_body, remaining_form.arguments) if remaining_form.arguments else replaced_body
    )


def _binding(
    entry: Function | Conjugate, value: AbstractExpression
) -> tuple[Function, AbstractExpression]:
    # The value is that of the entry, so v in an entry conjugate(v) takes the conjugate.
    match entry:
        case Conjugate(v):
            return _entry_function(entry), Conjugate(_promote(v, value))
        case _:
            return entry, _promote(entry, value)
