from __future__ import annotations

from collections.abc import Mapping
from typing import TypeAlias, cast, overload

from ..algorithms.rewriting import complete_family, promote
from ..algorithms.substitution import occurs, rename_bound, substitute
from ..nodes import (
    AbstractExpression,
    ComponentTensor,
    Conjugate,
    Form,
    Function,
    Index,
    IndexSum,
)
from ..nodes.arguments import argument_members
from ..nodes.forms import entry_function
from ..nodes.validation import admissible, merge_indices

Entry: TypeAlias = Function | Conjugate | tuple[Function | Conjugate, ...]
Value: TypeAlias = AbstractExpression | tuple[AbstractExpression, ...]


def partial(form: Form, mapping: Mapping[Entry, Value]) -> AbstractExpression:
    """Instantiate complete outer entries simultaneously; reject retained capture."""
    if not isinstance(cast("object", form), Form):
        raise TypeError("partial requires an explicit Form")
    if not mapping:
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
        indices = merge_indices(indices, *(m.free_indices for m in argument_members(value)))
        for member, supplied in zip(argument_members(entry), argument_members(value), strict=True):
            variable = entry_function(member)
            supplied = promote(variable, supplied)
            replacements[variable] = (
                Conjugate(supplied) if isinstance(member, Conjugate) else supplied
            )
    remaining = tuple(entry for entry in form.arguments if entry not in mapping)
    if any(
        occurs(entry_function(member), value)
        for entry in remaining
        for member in argument_members(entry)
        for value in replacements.values()
    ):
        raise ValueError("partial would capture a retained argument; rebind it explicitly first")
    body = cast("AbstractExpression", substitute(form.expression, replacements))
    body = complete_family(indices, body)
    return Form(body, remaining) if remaining else body


def _rebind_form(f: Form, associations: Mapping[Entry, Entry]) -> Form:
    """Rename complete outer form declarations without changing their spaces."""
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
def rebind(binder: Form, mapping: Mapping[Entry, Entry]) -> Form: ...


@overload
def rebind(binder: ComponentTensor, mapping: Mapping[Index, Index]) -> ComponentTensor: ...


@overload
def rebind(binder: IndexSum, mapping: Mapping[Index, Index]) -> IndexSum: ...


def rebind(
    binder: AbstractExpression, mapping: Mapping[Entry, Entry] | Mapping[Index, Index]
) -> Form | ComponentTensor | IndexSum:
    """Rename form entries or index declarations through the same binding rules."""
    if isinstance(binder, Form):
        return _rebind_form(binder, cast("Mapping[Entry, Entry]", mapping))
    if isinstance(binder, ComponentTensor | IndexSum):
        indices = cast("Mapping[Index, Index]", mapping)
        return rename_bound(binder, {old: new for old, new in indices.items()})
    raise TypeError("rebind requires an explicit Form, ComponentTensor or IndexSum")
