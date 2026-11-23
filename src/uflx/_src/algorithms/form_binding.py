from __future__ import annotations

from collections.abc import Mapping
from typing import TypeAlias, TypeVar, cast, overload

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
from ..nodes.arguments import (
    BindingEntry,
    FormEntry,
    argument_members,
    binding_name,
)
from ..nodes.forms import entry_function
from ..nodes.validation import admissible, merge_indices

_Entry = TypeVar("_Entry", bound=BindingEntry)
_IndexBinder = TypeVar("_IndexBinder", ComponentTensor, IndexSum)
Value: TypeAlias = AbstractExpression | tuple[AbstractExpression, ...]


def partial(form: Form, mapping: Mapping[FormEntry, Value]) -> AbstractExpression:
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


@overload
def rebind(binder: Form, mapping: Mapping[FormEntry, FormEntry]) -> Form: ...


@overload
def rebind(binder: _IndexBinder, mapping: Mapping[Index, Index]) -> _IndexBinder: ...


def rebind(
    binder: AbstractExpression, mapping: Mapping[_Entry, BindingEntry]
) -> Form | ComponentTensor | IndexSum:
    """Rename complete declarations simultaneously, preserving grouping and conjugation."""
    if not isinstance(binder, Form | ComponentTensor | IndexSum):
        raise TypeError("rebind requires an explicit Form, ComponentTensor or IndexSum")
    if not mapping:
        return binder
    entries = binder.binding_entries()
    renames: dict[Index | Function, Index | Function] = {}
    for source, target in mapping.items():
        if source not in entries:
            raise ValueError("unknown bound declaration")
        if isinstance(source, tuple) != isinstance(target, tuple):
            raise ValueError("rebind must preserve argument grouping")
        for old, new in zip(argument_members(source), argument_members(target), strict=True):
            if isinstance(old, Conjugate) != isinstance(new, Conjugate):
                raise ValueError("rebind must preserve conjugation")
            renames[binding_name(old)] = binding_name(new)
    return rename_bound(binder, renames)
