# Copyright (C) 2026 Jack S. Hale
#
# This file is part of uflx-core-experimental (https://fenics.github.io/uflx/language/)
#
# SPDX-License-Identifier:    MIT
"""Graph reconstruction and traversal of dependencies or expression values."""

from __future__ import annotations

from collections.abc import Callable, Generator, Hashable, Iterator, Sequence
from typing import Concatenate, ParamSpec, TypeVar, cast

from ..nodes import AbstractExpression, AbstractFunctionSpace, Base
from ..nodes.base import construction_equal, construction_field
from ..nodes.structure import value_operands
from ..typing import Argument
from ..typing.nodes import Construction

_P = ParamSpec("_P")
_R = TypeVar("_R")
_K = TypeVar("_K", bound=Hashable)


def rebuild(e: Base, **changes: Construction) -> Construction:
    # An unknown name adds an argument, which the class rejects.
    values = dict(zip(e.argument_names, e.construction_arguments)) | changes
    if tuple(values) == e.argument_names and construction_equal(
        tuple(values.values()), e.construction_arguments
    ):
        return e
    from ..nodes import ConjugateSpace, DualSpace
    from ..spaces import conjugate_space, dual_space

    if isinstance(e, ConjugateSpace):
        return conjugate_space(cast("AbstractFunctionSpace", values["space"]))
    if isinstance(e, DualSpace):
        return dual_space(cast("Argument[AbstractFunctionSpace]", values["space"]))
    return type(e)(*values.values())


def map_construction(value: Construction, fn: Callable[[Base], Construction]) -> Construction:
    if not isinstance(value, tuple):
        return fn(value) if isinstance(value, Base) else value
    # Each parent saves its position and its copy, allocated only on the first change.
    parents: list[tuple[tuple[Construction, ...], int, list[Construction] | None]] = []
    entries = value
    i = 0
    items: list[Construction] | None = None
    while True:
        if i == len(entries):
            mapped: Construction = entries if items is None else tuple(items)
            if len(parents) == 0:
                return mapped
            entries, i, items = parents.pop()
        else:
            item = entries[i]
            if isinstance(item, tuple):
                parents.append((entries, i, items))
                entries, i, items = item, 0, None
                continue
            mapped = fn(item) if isinstance(item, Base) else item
        if mapped is not entries[i]:
            if items is None:
                items = list(entries)
            items[i] = mapped
        i += 1


def map_fields(
    e: Base, fn: Callable[[Base], Construction], names: Sequence[str] | None = None
) -> Construction:
    """Apply fn to nodes in the named construction fields, or in all fields by default."""
    names = e.argument_names if names is None else names
    changes: dict[str, Construction] | None = None
    for name in names:
        value = construction_field(e, name)
        mapped = map_construction(value, fn)
        if changes is None:
            # An unknown field must still go through rebuild's constructor check.
            if mapped is value and name in e.argument_names:
                continue
            changes = {}
        changes[name] = mapped
    return e if changes is None else rebuild(e, **changes)


def memoized(rule: Callable[Concatenate[Callable[_P, _R], _P], _R]) -> Callable[_P, _R]:
    """Give rule a recursive visitor with a cache shared for the lifetime of the visitor."""
    memo: dict[tuple[object, ...], _R] = {}

    def recurse(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        key = (*args, *kwargs.items())
        if key not in memo:
            memo[key] = rule(recurse, *args, **kwargs)
        return memo[key]

    return recurse


def resolve(root: _K, visit: Callable[[_K], Generator[_K, None, _R]], done: dict[_K, _R]) -> _R:
    """Run a depth-first visit on an explicit stack.

    A visit yields dependency keys and returns its result. When it resumes,
    the requested result is in done. Requesting an active ancestor is a cycle.
    """
    if root in done:
        return done[root]

    def requests(key: _K) -> Iterator[_K]:
        done[key] = yield from visit(key)

    pending = [(root, requests(root))]
    active = {root}
    while len(pending) > 0:
        key, frame = pending[-1]
        try:
            child = next(frame)
        except StopIteration:
            _ = pending.pop()
            active.remove(key)
        else:
            if child not in done:
                if child in active:
                    raise ValueError("rewrite cycle", child)
                active.add(child)
                pending.append((child, requests(child)))
    return done[root]


def rewrite_values(
    root: AbstractExpression,
    rule: Callable[[AbstractExpression], AbstractExpression],
    complete: Callable[[AbstractExpression, AbstractExpression], AbstractExpression] | None = None,
) -> AbstractExpression:
    done: dict[AbstractExpression, AbstractExpression] = {}

    def visit(e: AbstractExpression) -> Generator[AbstractExpression, None, AbstractExpression]:
        changed = False
        for child in value_operands(e):
            yield child
            if done[child] is not child:
                changed = True
        rebuilt = map_values(e, done.__getitem__) if changed else e
        replacement = rule(rebuilt)
        if complete is None:
            return replacement
        if replacement != rebuilt:
            replacement = complete(rebuilt, replacement)
            if replacement != rebuilt:
                yield replacement
                return done[replacement]
        return rebuilt

    return resolve(root, visit, done)


def bottom_up(
    f: AbstractExpression, rewrite: Callable[[AbstractExpression], AbstractExpression]
) -> AbstractExpression:
    """Rewrite each distinct value node once, after visiting its operands."""
    return rewrite_values(f, rewrite)


def preorder(e: AbstractExpression) -> Iterator[AbstractExpression]:
    """Visit each distinct value node before its operands, in declaration order."""
    pending = [e]
    seen: set[AbstractExpression] = set()
    while len(pending) > 0:
        node = pending.pop()
        if node not in seen:
            seen.add(node)
            yield node
            pending.extend(reversed(value_operands(node)))


def everywhere(e: AbstractExpression, holds: Callable[[AbstractExpression], bool]) -> bool:
    return all(map(holds, preorder(e)))


def map_values(
    e: AbstractExpression,
    fn: Callable[[AbstractExpression], AbstractExpression],
    names: Sequence[str] | None = None,
) -> AbstractExpression:
    """Map fn over the immediate value operands, including those in tuple-valued fields."""
    fields = e.value_fields if names is None else names
    mapped = map_fields(e, lambda child: fn(cast("AbstractExpression", child)), fields)
    return cast("AbstractExpression", mapped)
