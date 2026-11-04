"""SB16-SB17. Apply explicit forms with capture-avoiding substitution."""

from __future__ import annotations

from typing import cast

from ..algorithms.rewriting import promote
from ..algorithms.substitution import freshen, substitute
from ..nodes import AbstractExpression, Conjugate, Form, Function
from ..nodes.arguments import argument_members


def bind(
    form: Form, position: int, argument: AbstractExpression | tuple[AbstractExpression, ...]
) -> AbstractExpression:
    remaining_arguments = form.arguments[:position] + form.arguments[position + 1 :]
    remaining_form = freshen(Form(form.expression, remaining_arguments), argument_members(argument))
    pairs = zip(argument_members(form.arguments[position]), argument_members(argument), strict=True)
    replacements: dict[Function, AbstractExpression] = {}
    for entry, value in pairs:
        if isinstance(entry, Conjugate):
            # Binding conjugate(v) to a value binds v to its conjugate.
            variable = cast("Function", entry.operand)
            replacements[variable] = Conjugate(promote(variable, value))
        else:
            replacements[entry] = promote(entry, value)
    replaced_body = cast("AbstractExpression", substitute(remaining_form.expression, replacements))
    return (
        Form(replaced_body, remaining_form.arguments) if remaining_form.arguments else replaced_body
    )
