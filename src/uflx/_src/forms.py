# Copyright (C) 2026 Jack S. Hale
#
# This file is part of uflx-core-experimental (https://fenics.github.io/uflx/language/)
#
# SPDX-License-Identifier:    MIT

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, cast

from .algorithms.rewriting import complete_family, promote
from .algorithms.substitution import occurs, substitute
from .nodes import (
    AbstractExpression,
    Apply,
    Conjugate,
    Form,
    Function,
)
from .nodes.arguments import argument_leaves, argument_pairs
from .nodes.forms import entry_function
from .nodes.validation import admissible, merge_indices
from .typing import Argument
from .typing.nodes import Construction, FormEntry

if TYPE_CHECKING:
    from .nodes.spaces import AbstractFunctionSpace


def function(space: AbstractFunctionSpace, label: str | tuple[Construction, ...]) -> Function:
    return Function(space, label)


def form(
    expression: AbstractExpression,
    arguments: tuple[FormEntry, ...],
) -> Form:
    return Form(expression, arguments)


def apply(
    form: AbstractExpression,
    arguments: tuple[Argument[AbstractExpression], ...],
) -> AbstractExpression:
    if arguments == () and form.argument_spaces == ():
        return form
    return Apply(form, arguments)


def partial(
    form: Form, mapping: Mapping[FormEntry, Argument[AbstractExpression]]
) -> AbstractExpression:
    """Instantiate complete outer entries simultaneously; reject retained capture."""
    if not isinstance(cast("object", form), Form):
        raise TypeError("partial requires an explicit Form")
    if len(mapping) == 0:
        return form
    if any(entry not in form.arguments for entry in mapping):
        raise ValueError("selector is not an outer argument entry")
    replacements: dict[Function, AbstractExpression] = {}
    indices = form.free_indices
    spaces = form.argument_spaces[len(form.expression.argument_spaces) :]
    for entry, space in zip(form.arguments, spaces, strict=True):
        if entry not in mapping:
            continue
        value = mapping[entry]
        admissible(space, value)
        indices = merge_indices(indices, *(m.free_indices for m in argument_leaves(value)))
        for member, supplied in argument_pairs(entry, value):
            variable = entry_function(member)
            supplied = promote(variable, supplied)
            replacements[variable] = (
                Conjugate(supplied) if isinstance(member, Conjugate) else supplied
            )
    remaining = tuple(entry for entry in form.arguments if entry not in mapping)
    if any(
        occurs(entry_function(member), value)
        for entry in remaining
        for member in argument_leaves(entry)
        for value in replacements.values()
    ):
        raise ValueError("partial would capture a retained argument; rebind it explicitly first")
    body = cast("AbstractExpression", substitute(form.expression, replacements))
    body = complete_family(indices, body)
    return Form(body, remaining) if remaining != () else body
