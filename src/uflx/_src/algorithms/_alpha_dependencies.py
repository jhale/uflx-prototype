# Copyright (C) 2026 Jack S. Hale
#
# This file is part of uflx-core-experimental (https://fenics.github.io/uflx/language/)
#
# SPDX-License-Identifier:    MIT
"""Persistent sets for lexical dependencies, shared across source DAG nodes.

Unlike a frozenset per subtree, adding one name copies only a treap search path.
Union uses treap splitting and shares untouched branches. Priorities are derived
from locally assigned name IDs: balancing is heuristic, not a worst-case bound.
"""

from __future__ import annotations

from dataclasses import dataclass

from ._alpha_summary import payload_digest as digest


@dataclass(frozen=True, eq=False, slots=True)
class Dependency:
    key: int
    priority: int
    left: Dependency | None
    right: Dependency | None


def node(key: int, priority: int, left: Dependency | None, right: Dependency | None) -> Dependency:
    return Dependency(key, priority, left, right)


def singleton(key: int) -> Dependency:
    """Return the immutable singleton for a caller-assigned name key."""
    return node(key, digest(("dependency", key)), None, None)


def rebuild(tree: Dependency, left: Dependency | None, right: Dependency | None) -> Dependency:
    if left is tree.left and right is tree.right:
        return tree
    return node(tree.key, tree.priority, left, right)


def split(tree: Dependency | None, key: int) -> tuple[Dependency | None, Dependency | None]:
    """Return keys below and above key, dropping key itself if present."""
    if tree is None:
        return None, None
    if tree.key == key:
        return tree.left, tree.right
    if tree.key < key:
        left, right = split(tree.right, key)
        return rebuild(tree, tree.left, left), right
    left, right = split(tree.left, key)
    return left, rebuild(tree, right, tree.right)


def union(a: Dependency | None, b: Dependency | None) -> Dependency | None:
    if a is b or b is None:
        return a
    if a is None:
        return b
    if (a.priority, a.key) < (b.priority, b.key):
        a, b = b, a
    left, right = split(b, a.key)
    return rebuild(a, union(a.left, left), union(a.right, right))


def join(a: Dependency | None, b: Dependency | None) -> Dependency | None:
    """Join sets whose keys are strictly ordered across the boundary."""
    if a is None:
        return b
    if b is None:
        return a
    if (a.priority, a.key) > (b.priority, b.key):
        return rebuild(a, a.left, join(a.right, b))
    return rebuild(b, join(a, b.left), b.right)


def erase(tree: Dependency | None, key: int) -> Dependency | None:
    if tree is None:
        return None
    if tree.key == key:
        return join(tree.left, tree.right)
    if key < tree.key:
        return rebuild(tree, erase(tree.left, key), tree.right)
    return rebuild(tree, tree.left, erase(tree.right, key))


def without(tree: Dependency | None, keys: tuple[int, ...]) -> Dependency | None:
    """Remove keys, sharing branches unaffected by the removals."""
    for key in keys:
        tree = erase(tree, key)
    return tree


def contains(tree: Dependency | None, key: int) -> bool:
    """Test exact membership without changing the set."""
    while tree is not None:
        if tree.key == key:
            return True
        tree = tree.left if key < tree.key else tree.right
    return False
