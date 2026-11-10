# SPDX-License-Identifier: MIT
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
    size: int


class Dependencies:
    """One adapter's arena of persistent sets and diagnostic work counters."""

    def __init__(self) -> None:
        self.names: dict[object, Dependency] = {}
        self.steps = 0
        self.created = 0

    def node(
        self, key: int, priority: int, left: Dependency | None, right: Dependency | None
    ) -> Dependency:
        self.created += 1
        return Dependency(
            key,
            priority,
            left,
            right,
            1 + (left.size if left else 0) + (right.size if right else 0),
        )

    def singleton(self, name: object) -> Dependency:
        if name not in self.names:
            key = len(self.names)
            self.names[name] = self.node(key, digest(("dependency", key)), None, None)
        return self.names[name]

    def rebuild(
        self, tree: Dependency, left: Dependency | None, right: Dependency | None
    ) -> Dependency:
        if left is tree.left and right is tree.right:
            return tree
        return self.node(tree.key, tree.priority, left, right)

    def split(
        self, tree: Dependency | None, key: int
    ) -> tuple[Dependency | None, Dependency | None]:
        """Return keys below and above key, dropping key itself if present."""
        self.steps += 1
        if tree is None:
            return None, None
        if tree.key == key:
            return tree.left, tree.right
        if tree.key < key:
            left, right = self.split(tree.right, key)
            return self.rebuild(tree, tree.left, left), right
        left, right = self.split(tree.left, key)
        return left, self.rebuild(tree, right, tree.right)

    def union(self, a: Dependency | None, b: Dependency | None) -> Dependency | None:
        self.steps += 1
        if a is b or b is None:
            return a
        if a is None:
            return b
        if (a.priority, a.key) < (b.priority, b.key):
            a, b = b, a
        left, right = self.split(b, a.key)
        return self.rebuild(a, self.union(a.left, left), self.union(a.right, right))

    def join(self, a: Dependency | None, b: Dependency | None) -> Dependency | None:
        """Join sets whose keys are strictly ordered across the boundary."""
        self.steps += 1
        if a is None:
            return b
        if b is None:
            return a
        if (a.priority, a.key) > (b.priority, b.key):
            return self.rebuild(a, a.left, self.join(a.right, b))
        return self.rebuild(b, self.join(a, b.left), b.right)

    def erase(self, tree: Dependency | None, key: int) -> Dependency | None:
        self.steps += 1
        if tree is None:
            return None
        if tree.key == key:
            return self.join(tree.left, tree.right)
        if key < tree.key:
            return self.rebuild(tree, self.erase(tree.left, key), tree.right)
        return self.rebuild(tree, tree.left, self.erase(tree.right, key))

    def without(self, tree: Dependency | None, names: tuple[object, ...]) -> Dependency | None:
        for name in names:
            if name in self.names:
                tree = self.erase(tree, self.names[name].key)
        return tree

    def contains(self, tree: Dependency | None, name: object) -> bool:
        entry = self.names.get(name)
        if entry is None:
            return False
        while tree is not None:
            self.steps += 1
            if tree.key == entry.key:
                return True
            tree = tree.left if entry.key < tree.key else tree.right
        return False
