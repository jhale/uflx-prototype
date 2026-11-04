"""SB16-SB17. Apply explicit forms with capture-avoiding substitution."""

from __future__ import annotations

from typing import cast

from ..nodes import AbstractExpression, Conjugate, Form, Function
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
    replacements: dict[Function, AbstractExpression] = {}
    for entry, value in pairs:
        if isinstance(entry, Conjugate):
            # Binding conjugate(v) to a value binds v to its conjugate.
            variable = cast("Function", entry.operand)
            replacements[variable] = Conjugate(_promote(variable, value))
        else:
            replacements[entry] = _promote(entry, value)
    replaced_body = cast("AbstractExpression", _substitute(remaining_form.expression, replacements))
    return (
        Form(replaced_body, remaining_form.arguments) if remaining_form.arguments else replaced_body
    )
