# Copyright (C) 2026 Jack S. Hale
#
# This file is part of uflx-core-experimental (https://fenics.github.io/uflx/language/)
#
# SPDX-License-Identifier:    MIT

from __future__ import annotations

import weakref
from collections.abc import Callable, Generator, Iterable, Mapping, Sequence
from functools import cache
from typing import TYPE_CHECKING, TypeVar, cast, overload

from ..nodes import (
    AbstractExpression,
    Base,
    ComponentTensor,
    Conjugate,
    Form,
    Function,
    Index,
    IndexSum,
    MappedSpace,
    ReferenceValue,
    Zero,
)
from ..nodes.arguments import argument_pairs, binding_name
from ..nodes.base import construction_field
from ..nodes.structure import construction_nodes, operands
from ..typing.nodes import BindingEntry, Construction, FormEntry, Renames
from .binding import binding_signature, scoped_body_fields
from .traversal import map_construction, map_fields, rebuild, resolve

if TYPE_CHECKING:
    from ..nodes.indices import AbstractExtent


_T = TypeVar("_T", bound=Base)
_K = TypeVar("_K", bound=Base)
_Entry = TypeVar("_Entry", bound="BindingEntry")
_IndexBinder = TypeVar("_IndexBinder", ComponentTensor, IndexSum)
_Match = tuple[Base, frozenset[Base]]
_Request = tuple[Base, frozenset[Base], Renames]


def occurs(target: Base, node: Construction) -> bool:
    """Test free occurrence through construction dependencies, excluding labels."""
    return _occurs(target, node, True)


def occurs_in_values(target: Base, node: Construction) -> bool:
    """Test free occurrence in values, holding domains and spaces fixed."""
    return _occurs(target, node, False)


def occurs_in_metadata(target: Base, node: Construction) -> bool:
    """Detect dependencies outside value/index operands, including nested metadata."""

    pending = list(construction_nodes(node))
    seen: set[Base] = set()
    while len(pending) > 0:
        current = pending.pop()
        if current in seen:
            continue
        seen.add(current)
        # A mapped function's reference coefficients are held fixed under mesh motion.
        if (
            isinstance(current, ReferenceValue)
            and isinstance(current.expression, Function)
            and current.expression.space == current.space
        ):
            if occurs(target, current.space.reference_space):
                return True
            continue
        signature = tuple(
            s.reference_space if isinstance(s, MappedSpace) else s
            for s in binding_signature(current)
        )
        if occurs(target, signature):
            return True
        for field in _scoped(current, True, _bound_in(target, current)):
            operand = construction_field(current, field)
            if field in (*current.value_fields, *current.index_fields):
                pending.extend(construction_nodes(operand))
            elif occurs(target, operand):
                return True
    return False


def _occurs(target: Base, node: Construction, dependencies: bool) -> bool:
    cache_name = "occurrences" if dependencies else "value_occurrences"
    done: dict[tuple[Base, Base], bool] = {}

    def visit(query: tuple[Base, Base]) -> Generator[tuple[Base, Base], None, bool]:
        needle, current = query
        memo = cast("weakref.WeakKeyDictionary[Base, bool] | None", vars(current).get(cache_name))
        if memo is None:
            memo = weakref.WeakKeyDictionary[Base, bool]()
            vars(current)[cache_name] = memo
        elif needle in memo:
            return memo[needle]

        found = current == needle
        if not found and dependencies:
            for child in construction_nodes(binding_signature(current)):
                yield needle, child
                if done[needle, child]:
                    found = True
                    break
        if not found:
            shadowed = False
            for name in current.bound():
                yield name, needle
                if done[name, needle]:
                    shadowed = True
                    break
            for child in operands(current, _scoped(current, dependencies, shadowed)):
                yield needle, child
                if done[needle, child]:
                    found = True
                    break
        memo[needle] = found
        return found

    return any(resolve((target, child), visit, done) for child in construction_nodes(node))


def _scoped(node: Base, dependencies: bool, shadowed: bool) -> list[str]:
    fields = node.argument_names if dependencies else (*node.value_fields, *node.index_fields)
    return [
        name
        for name in fields
        if name not in (*node.labels, *node.binding_fields)
        and not (shadowed and name in node.scope)
    ]


def _bound_in(target: Base, node: Base, dependencies: bool = True) -> bool:
    """Whether the target contains a name bound on entering this scope."""
    return any(_occurs(name, target, dependencies) for name in node.bound())


def _rename(node: Base, name: Index | Function, avoid: Sequence[Construction]) -> Index | Function:
    # The label refers to the bound operands and to avoid, never to the binding node.
    label = _renamed_label(node, name, avoid)
    if isinstance(name, Index):
        return type(name)(label)
    return Function(name.space, label)


def _fresh_names(node: Base, avoid: Sequence[Construction]) -> Renames:
    return tuple(
        (name, _rename(node, name, avoid))
        for name in node.bound()
        if any(occurs(name, replacement) for replacement in avoid)
    )


def freshen(node: _T, avoid: Sequence[Construction]) -> _T:
    renames = _fresh_names(node, avoid)
    return _rename_scope(node, dict(renames)) if renames != () else node


def _rename_scope(node: _T, renames: Mapping[_K, Construction]) -> _T:
    return cast("_T", map_fields(node, lambda child: substitute(child, renames), node.scope))


def rename_bound(node: _T, renames: Mapping[Index | Function, Index | Function]) -> _T:
    """Rename declarations simultaneously, preserving type and lexical scope.

    rebind validates entry grouping and conjugation. This operation
    handles names uniformly, including derivative variables whose evaluation
    point and direction lie outside their scope.
    """
    names = node.bound()
    for old, new in renames.items():
        if old not in names:
            raise ValueError("unknown bound name")
        if type(old) is not type(new):
            raise ValueError("renaming must preserve the kind of bound name")
        if isinstance(old, Function) and isinstance(new, Function) and old.space != new.space:
            raise ValueError("renaming must preserve parameter spaces")
        if new not in names and any(
            occurs(new, child) for child in operands(node, scoped_body_fields(node))
        ):
            raise ValueError("renaming would capture free data")
    targets = tuple(renames.get(name, name) for name in names)
    if len(set(targets)) != len(targets):
        raise ValueError("renaming would create duplicate bound names")
    signature = binding_signature(node)
    if any(occurs(name, signature) for name in (*names, *targets)):
        raise ValueError("renaming requires independent parameter types")
    result = _rename_scope(node, renames)
    if binding_signature(result) != signature:
        raise ValueError("renaming changed the binding signature")
    return result


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
    if len(mapping) == 0:
        return binder
    entries = binder.binding_entries()
    renames: dict[Index | Function, Index | Function] = {}
    declarations: Iterable[tuple[BindingEntry, BindingEntry]] = mapping.items()
    for source, target in declarations:
        if source not in entries:
            raise ValueError("unknown bound declaration")
        for old, new in argument_pairs(source, target):
            if isinstance(old, Conjugate) != isinstance(new, Conjugate):
                raise ValueError("rebind must preserve conjugation")
            renames[binding_name(old)] = binding_name(new)
    return rename_bound(binder, renames)


def substitute(node: Base, mapping: Mapping[_K, Construction]) -> Construction:
    return _substitute(node, mapping, dependencies=True)


def _substitute_zero(node: Zero, replace: Callable[[Base], Construction]) -> Construction:
    from ..nodes import GatheredIndex
    from ..nodes.validation import entries, merge_indices

    # Zero's free-index signature changes when an index is selected or gathered.
    pairs: list[tuple[tuple[Index, int | AbstractExtent], ...]] = []
    for i, extent in node.free_indices:
        entry = replace(i)
        size = cast("int | AbstractExtent", replace(extent)) if isinstance(extent, Base) else extent
        if not (type(entry) is int or isinstance(entry, Index | GatheredIndex)):
            raise TypeError("replacement of an index must be an index entry")
        pairs.append(entries((size,), (entry,)))
    new = map_fields(node, replace, ("value_shape", "argument_spaces", "scalar_type"))
    return rebuild(cast("Base", new), free_indices=merge_indices(*pairs))


def _substitute(
    root: Base, replacements: Mapping[_K, Construction], *, dependencies: bool
) -> Construction:
    """Match on the original graph and rebuild with capture-avoiding binder names."""
    mapping: dict[Base, Construction] = dict(
        sorted(replacements.items(), key=lambda pair: pair[0].digest)
    )
    empty: frozenset[Base] = frozenset()

    def fields(node: Base) -> tuple[str, ...]:
        if dependencies or isinstance(node, Zero):
            return node.argument_names
        if node.labels != ():
            return ()
        return (*node.value_fields, *node.index_fields)

    @cache
    def eligible(node: Base, keys: frozenset[Base]) -> frozenset[Base]:
        return frozenset(key for key in keys if not _bound_in(key, node))

    matched: dict[_Match, frozenset[Base]] = {}

    def matches(query: _Match) -> Generator[_Match, None, frozenset[Base]]:
        # A whole-node match masks its children. Label contents never add free occurrences.
        node, keys = query
        if node in keys:
            return frozenset((node,))
        found: set[Base] = set()
        for field in fields(node):
            active = eligible(node, keys) if field in node.scope else keys
            if len(active) > 0 and field not in node.labels:
                for child in operands(node, (field,)):
                    yield child, active
                    found.update(matched[child, active])
        return frozenset(found)

    def scope_renames(node: Base, keys: frozenset[Base], renames: Renames) -> Renames:
        inherited = tuple(pair for pair in renames if not _bound_in(pair[0], node))
        if not any(occurs(name, mapping[key]) for name in node.bound() for key in keys):
            return inherited
        scoped = tuple(field for field in fields(node) if field in node.scope)
        used = {
            key
            for child in operands(node, scoped)
            for key in resolve((child, keys), matches, matched)
        }
        avoid = tuple(value for key, value in mapping.items() if key in used)
        return inherited + _fresh_names(node, avoid)

    done: dict[_Request, Construction] = {}

    def visit(request: _Request) -> Generator[_Request, None, Construction]:
        node, keys, renames = request
        # Match the original node before applying any enclosing binder's renames.
        if node in keys:
            return mapping[node]
        if len(keys) == 0 and renames == ():
            return node
        if isinstance(node, Zero):
            for child in operands(node):
                yield child, keys, renames
            return _substitute_zero(node, lambda child: done[child, keys, renames])

        selected = fields(node)
        inner_keys, inner_renames = keys, renames
        if node.scope != ():
            inner_keys = eligible(node, keys)
            inner_renames = scope_renames(node, inner_keys, renames)

        renamed = dict(renames).get(node)
        changes: dict[str, Construction] = {}
        for field in node.argument_names:
            if field in node.labels:
                if renamed is not None:
                    changes[field] = renamed.label
                    continue
                if _is_renamed_label(construction_field(node, field)):
                    continue
            active, bound = (inner_keys, inner_renames) if field in node.scope else (keys, renames)
            if field not in selected:
                active = empty
            if len(active) > 0 or bound != ():
                value = construction_field(node, field)
                if isinstance(value, Base):
                    yield value, active, bound
                    changes[field] = done[value, active, bound]
                else:
                    for child in construction_nodes(value):
                        yield child, active, bound
                    changes[field] = map_construction(
                        value, lambda child: done[child, active, bound]
                    )
        return rebuild(node, **changes)

    return resolve((root, frozenset(mapping), ()), visit, done)


def replace(e: Base, mapping: dict[Base, Construction]) -> Construction:
    """Substitute simultaneously through construction dependencies, respecting binding."""
    return substitute(e, mapping)


def substitute_values(
    node: AbstractExpression, mapping: Mapping[_K, Construction]
) -> AbstractExpression:
    """Substitute through expression values, preserving binding, geometry and identity metadata."""
    return cast("AbstractExpression", _substitute(node, mapping, dependencies=False))


def _renamed_label(
    node: Base, name: Base, avoid: Sequence[Construction]
) -> tuple[Construction, ...]:
    """Record a capture-avoidance identity without referring to its binding node.

    The scoped operands, original name and ordered avoidance context distinguish
    this renaming. Substitution must subsequently treat the label as fixed.
    """
    body = tuple(construction_field(node, field) for field in node.scope)
    return ("uflx.rename", name, body, tuple(avoid))


def _is_renamed_label(label: Construction) -> bool:
    """Whether a label records a fixed capture-avoidance identity."""
    return isinstance(label, tuple) and label[:1] == ("uflx.rename",)
