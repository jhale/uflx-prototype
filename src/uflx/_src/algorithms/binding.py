# Copyright (C) 2026 Jack S. Hale
#
# This file is part of uflx-core-experimental (https://fenics.github.io/uflx/language/)
#
# SPDX-License-Identifier:    MIT
"""Lexical binding shared by function, index and derivative declarations."""

from __future__ import annotations

from collections.abc import Iterator
from typing import NamedTuple

from ..nodes import AbstractExpression, Base, Function, Index
from ..nodes.base import construction_equal, construction_field
from ..nodes.structure import operands
from ..typing.nodes import Bindings, Construction, Names


class _Comparison(NamedTuple):
    left: Construction
    right: Construction
    left_bindings: Bindings
    right_bindings: Bindings


def scoped_body_fields(node: Base) -> tuple[str, ...]:
    """Fields in the lexical scope, excluding declarations themselves."""
    return tuple(field for field in node.scope if field not in node.binding_fields)


def binding_signature(node: Base) -> tuple[Construction, ...]:
    """Ordered function spaces or index extents, independent of binder labels."""
    names = node.bound()
    if names == ():
        return ()
    if all(isinstance(name, Function) for name in names):
        return tuple(name.space for name in names if isinstance(name, Function))
    extents = {
        index: extent
        for body in operands(node, scoped_body_fields(node))
        if isinstance(body, AbstractExpression)
        for index, extent in body.free_indices
    }
    return tuple(name.space if isinstance(name, Function) else extents[name] for name in names)


def _enter(bindings: Bindings, names: Names) -> Bindings:
    return tuple(
        (name, (depth + 1, slot)) for name, (depth, slot) in bindings if name not in names
    ) + tuple((name, (0, slot)) for slot, name in enumerate(names))


def _address(name: Function | Index, bindings: Bindings) -> tuple[int, int] | None:
    return next((address for bound, address in bindings if bound == name), None)


def alpha_equivalent(left: Base, right: Base) -> bool:
    """Compare lexical declarations without changing structural equality.

    Labels retain structural identity. Spaces and other construction metadata
    are compared under the enclosing bindings, including parameter signatures
    outside their own binder. No algebraic or type normalisation is performed.

    Cache keys retain only bindings relevant to each subtree, preserving their
    lexical addresses. Thus irrelevant enclosing scopes do not multiply work on
    shared nodes. Ordinary node equality, hashing and interning are unchanged.
    """

    if left is right:
        return True

    dependency_cache: dict[Base, frozenset[Function | Index]] = {}
    signatures: dict[Base, tuple[Construction, ...]] = {}

    def dependencies(value: Construction) -> frozenset[Function | Index]:
        return frozenset[Function | Index]().union(
            *(dependency_cache[node] for node in _construction_nodes(value))
        )

    def node_dependencies(root: Base) -> frozenset[Function | Index]:
        # Postorder traversal: compute each node only after its dependencies.
        if root in dependency_cache:
            return dependency_cache[root]
        stack = [(root, False)]
        while len(stack) > 0:
            node, ready = stack.pop()
            if node in dependency_cache:
                continue
            fields = tuple(field for field in node.argument_names if field not in node.labels)
            if not ready:
                signature = signatures.setdefault(node, binding_signature(node))
                stack.append((node, True))
                children = list(_construction_nodes(signature))
                for field in fields:
                    children.extend(_construction_nodes(construction_field(node, field)))
                stack.extend((child, False) for child in children if child not in dependency_cache)
                continue
            names = node.bound()
            result = set(dependencies(signatures[node]))
            if isinstance(node, Function | Index):
                result.add(node)
            for field in fields:
                child = dependencies(construction_field(node, field))
                result.update(child.difference(names) if field in node.scope else child)
            dependency_cache[node] = frozenset(result)
        return dependency_cache[root]

    def project(node: Base, bindings: Bindings) -> Bindings:
        relevant = node_dependencies(node)
        return tuple(pair for pair in bindings if pair[0] in relevant)

    pending = [_Comparison(left, right, (), ())]
    seen: set[tuple[Base, Base, Bindings, Bindings]] = set()

    def compare_nodes(a: Base, b: Base, sa: Bindings, sb: Bindings) -> bool:
        if a is b and sa == sb:
            return True
        sa, sb = project(a, sa), project(b, sb)
        if a is b and sa == sb:
            return True
        key = (a, b, sa, sb)
        if key in seen:
            return True
        seen.add(key)
        if isinstance(a, Function | Index) and isinstance(b, Function | Index):
            aa, ab = _address(a, sa), _address(b, sb)
            if aa is not None or ab is not None:
                return aa == ab
            # Free functions retain their labels, but their spaces may depend
            # on a name bound outside this node. Compare those fields below.
        na, nb = a.bound(), b.bound()
        if len(na) != len(nb):
            return False
        inner_a, inner_b = (_enter(sa, na), _enter(sb, nb)) if na != () else (sa, sb)
        for field in reversed(a.argument_names):
            x, y = construction_field(a, field), construction_field(b, field)
            if field in a.labels:
                if not construction_equal(x, y):
                    return False
            else:
                pending.append(
                    _Comparison(
                        x,
                        y,
                        inner_a if field in a.scope else sa,
                        inner_b if field in b.scope else sb,
                    )
                )
        pending.append(_Comparison(signatures[a], signatures[b], sa, sb))
        return True

    while len(pending) > 0:
        a, b, sa, sb = pending.pop()
        if type(a) is not type(b):
            return False
        if isinstance(a, tuple) and isinstance(b, tuple):
            if len(a) != len(b):
                return False
            pending.extend(
                _Comparison(x, y, sa, sb) for x, y in zip(reversed(a), reversed(b), strict=True)
            )
        elif isinstance(a, Base) and isinstance(b, Base):
            if not compare_nodes(a, b, sa, sb):
                return False
        elif not construction_equal(a, b):
            return False
    return True


def construction_nodes(value: Construction) -> Iterator[Base]:
    """Iterate node dependencies through tuples without traversing node fields."""
    return _construction_nodes(value)


def _construction_nodes(value: Construction) -> Iterator[Base]:
    """Iterate immediate node dependencies through arbitrarily nested tuples."""
    pending = [value]
    while len(pending) > 0:
        item = pending.pop()
        if isinstance(item, Base):
            yield item
        elif isinstance(item, tuple):
            pending.extend(reversed(item))
