"""Experimental named binding API; not yet part of the language specification."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TypeAlias, cast, overload

from ..algorithms.binding import alpha_equivalent as alpha_equivalent
from ..algorithms.rewriting import promote
from ..algorithms.substitution import occurs, rename_bound, substitute
from ..nodes import (
    AbstractExpression,
    Action,
    ComponentTensor,
    Conjugate,
    Form,
    Function,
    Index,
    IndexSum,
)
from ..nodes.arguments import argument_members
from ..nodes.forms import entry_function

Entry: TypeAlias = Function | Conjugate | tuple[Function | Conjugate, ...]
Value: TypeAlias = AbstractExpression | tuple[AbstractExpression, ...]


def _independent_parameters(f: Form) -> None:
    if any(occurs(name, f.argument_spaces) for name in f.bound()):
        raise ValueError("dependent parameter spaces are unsupported by this binding prototype")


def apply(f: AbstractExpression, arguments: tuple[Value, ...]) -> AbstractExpression:
    """Apply all ordered slots, retaining lazy Action nodes internally."""
    if type(arguments) is not tuple:
        raise TypeError("application requires an ordered tuple")
    if len(arguments) != len(f.argument_spaces):
        raise ValueError("application must supply every argument slot")
    for argument in arguments:
        f = Action(f, 0, argument)
    return f


def partial(f: AbstractExpression, associations: Mapping[Entry, Value]) -> AbstractExpression:
    """Instantiate complete outer entries simultaneously; reject retained capture."""
    if not isinstance(f, Form):
        raise TypeError("partial requires an explicit Form")
    _independent_parameters(f)
    if not associations:
        return f
    replacements: dict[Function, AbstractExpression] = {}
    for entry, value in associations.items():
        if entry not in f.arguments:
            raise ValueError("selector is not an outer argument entry")
        Action(f, len(f.expression.argument_spaces) + f.arguments.index(entry), value)
        for member, supplied in zip(argument_members(entry), argument_members(value), strict=True):
            variable = entry_function(member)
            supplied = promote(variable, supplied)
            replacements[variable] = (
                Conjugate(supplied) if isinstance(member, Conjugate) else supplied
            )
    remaining = tuple(entry for entry in f.arguments if entry not in associations)
    retained = Form(f.expression, remaining)
    if any(occurs(name, value) for name in retained.bound() for value in replacements.values()):
        raise ValueError("partial would capture a retained argument; rebind it explicitly first")
    body = cast("AbstractExpression", substitute(f.expression, replacements))
    return Form(body, remaining) if remaining else body


def _rebind_form(f: Form, associations: Mapping[Entry, Entry]) -> Form:
    """Rename complete outer form declarations without changing their spaces."""
    _independent_parameters(f)
    renames: dict[Index | Function, Index | Function] = {}
    for source, target in associations.items():
        if source not in f.arguments:
            raise ValueError("selector is not an outer argument entry")
        if isinstance(source, tuple) != isinstance(target, tuple):
            raise ValueError("rebind must preserve argument grouping")
        for old, new in zip(argument_members(source), argument_members(target), strict=True):
            if isinstance(old, Conjugate) != isinstance(new, Conjugate):
                raise ValueError("rebind must preserve conjugation")
            old_name, new_name = entry_function(old), entry_function(new)
            if old_name.space != new_name.space:
                raise ValueError("rebind must preserve parameter spaces")
            renames[old_name] = new_name
    return rename_bound(f, renames)


@overload
def rebind(f: Form, associations: Mapping[Entry, Entry]) -> Form: ...


@overload
def rebind(f: ComponentTensor, associations: Mapping[Index, Index]) -> ComponentTensor: ...


@overload
def rebind(f: IndexSum, associations: Mapping[Index, Index]) -> IndexSum: ...


def rebind(
    f: AbstractExpression, associations: Mapping[Entry, Entry] | Mapping[Index, Index]
) -> Form | ComponentTensor | IndexSum:
    """Rename form entries or index declarations through the same binding rules."""
    if isinstance(f, Form):
        return _rebind_form(f, cast("Mapping[Entry, Entry]", associations))
    if isinstance(f, ComponentTensor | IndexSum):
        indices = cast("Mapping[Index, Index]", associations)
        return rename_bound(f, {old: new for old, new in indices.items()})
    raise TypeError("rebind requires an explicit Form, ComponentTensor or IndexSum")
