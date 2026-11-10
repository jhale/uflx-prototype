# SPDX-License-Identifier: MIT
"""Deterministic fingerprints modulo the interpreter's lexical alpha equivalence.

UFLx-specific adapter to the reference algorithm port in _alpha_summary.py,
which contains the paper citation and upstream implementation's license notice.

Labels retain exact construction identity; parameter signatures are outside their
own binder. Dependency sets share unchanged branches. Context projection still
scans enclosing bindings: no whole-adapter complexity bound is claimed.
"""

from __future__ import annotations

from typing import NamedTuple, TypeAlias

from ..nodes import Base, Function, Index
from ..nodes.base import construction_encoding
from ..nodes.types import Construction
from ._alpha_dependencies import Dependencies, Dependency
from ._alpha_summary import Hasher, Summary
from .binding import binding_signature, construction_nodes

_Environment: TypeAlias = tuple[tuple[Function | Index, int], ...]
_Child: TypeAlias = tuple[Construction, _Environment]
_Key: TypeAlias = tuple[object, ...]


class _Recipe(NamedTuple):
    tag: object
    children: tuple[_Child, ...]
    tokens: tuple[int, ...] | None


class AlphaHasher:
    """Reuse alpha summaries across expressions within one compiler pass.

    The cache holds strong references to graphs. Discard this object or call
    clear() when the pass finishes. Fingerprints are deterministic across processes
    for the same construction data, but are not a versioned persistent-cache format
    or a cryptographic digest. Always confirm matches with alpha_equivalent.

    Each expression is compared in isolation: free identities remain significant.
    This does not identify open bodies under arbitrary enclosing binders.
    """

    def __init__(self) -> None:
        self._backend = Hasher()
        self._dependencies: dict[Base, Dependency | None] = {}
        self._dependency_sets = Dependencies()
        self._signatures: dict[Base, tuple[Construction, ...]] = {}
        self._memo: dict[_Key, Summary] = {}
        self._keep: dict[int, tuple[Construction, ...]] = {}
        self._tokens = 0
        self._visits = 0

    def clear(self) -> None:
        """Release cached graphs and summaries; the hasher can then be reused."""
        self.__init__()

    def fingerprint(self, expression: Base) -> int:
        """Return a candidate key invariant under lexical renaming of bound names."""
        return self._summary(expression).fingerprint

    def _value_dependencies(self, value: Construction) -> Dependency | None:
        result = None
        for node in construction_nodes(value):
            result = self._dependency_sets.union(result, self._dependencies[node])
        return result

    def _prepare(self, root: Base) -> None:
        pending = [(root, False)]
        while pending:
            node, ready = pending.pop()
            if node in self._dependencies:
                continue
            fields = tuple(f for f in node.argument_names if f not in node.labels)
            if not ready:
                signature = self._signatures.setdefault(node, binding_signature(node))
                pending.append((node, True))
                children = list(construction_nodes(signature))
                for field in fields:
                    children.extend(construction_nodes(getattr(node, field)))
                pending.extend(
                    (child, False) for child in children if child not in self._dependencies
                )
                continue
            names = node.bound()
            sets = self._dependency_sets
            deps = self._value_dependencies(self._signatures[node])
            if isinstance(node, Function | Index):
                deps = sets.union(deps, sets.singleton(node))
            for field in fields:
                child = self._value_dependencies(getattr(node, field))
                deps = sets.union(
                    deps, sets.without(child, names) if field in node.scope else child
                )
            self._dependencies[node] = deps

    def _context(self, value: Construction, environment: _Environment) -> _Environment:
        if isinstance(value, Base):
            return tuple(
                pair
                for pair in environment
                if self._dependency_sets.contains(self._dependencies[value], pair[0])
            )
        return environment

    def _key(self, value: Construction, environment: _Environment) -> _Key:
        if isinstance(value, Base):
            return ("node", value, self._context(value, environment))
        if isinstance(value, tuple):
            # Python conflates numeric types in tuple equality. Retaining the
            # tuple also prevents reuse of its id during this hasher's lifetime.
            self._keep[id(value)] = value
            return ("tuple", id(value), environment)
        return ("literal", construction_encoding(value))

    def _summary(self, root: Base) -> Summary:
        self._prepare(root)
        backend = self._backend
        pending: list[tuple[Construction, _Environment, _Recipe | None]] = [(root, (), None)]
        while pending:
            value, environment, recipe = pending.pop()
            environment = self._context(value, environment)
            key = self._key(value, environment)
            if key in self._memo:
                continue
            if recipe is not None:
                results = tuple(self._memo[self._key(child, env)] for child, env in recipe.children)
                if recipe.tokens is None:
                    result = backend.constructor(recipe.tag, results)
                else:
                    assert isinstance(value, Base)
                    signature, *fields = results
                    inside = backend.constructor(
                        ("scoped",),
                        tuple(
                            result
                            for field, result in zip(value.argument_names, fields, strict=True)
                            if field in value.scope
                        ),
                    )
                    outside = tuple(
                        result
                        for field, result in zip(value.argument_names, fields, strict=True)
                        if field not in value.scope
                    )
                    result = backend.constructor(
                        recipe.tag, (signature, backend.binder(recipe.tokens, inside), *outside)
                    )
                self._memo[key] = result
                self._visits += 1
                continue
            if isinstance(value, Function | Index):
                token = next((token for name, token in environment if name == value), None)
                if token is not None:
                    self._memo[key] = backend.constructor(
                        ("bound-type", value.identifier), (backend.variable(token),)
                    )
                    continue
            if isinstance(value, tuple):
                recipe = _Recipe(("tuple",), tuple((child, environment) for child in value), None)
            elif isinstance(value, Base):
                names = value.bound()
                tokens = tuple(range(self._tokens, self._tokens + len(names)))
                self._tokens += len(names)
                inner = tuple(pair for pair in environment if pair[0] not in names) + tuple(
                    zip(names, tokens, strict=True)
                )
                children: list[_Child] = [(self._signatures[value], environment)]
                for field in value.argument_names:
                    child: Construction = getattr(value, field)
                    if field in value.labels:
                        child = ("opaque", construction_encoding(child))
                    children.append((child, inner if field in value.scope else environment))
                recipe = _Recipe(
                    ("node", value.identifier, value.argument_names, value.scope),
                    tuple(children),
                    tokens,
                )
            else:
                self._memo[key] = backend.constructor(("literal", construction_encoding(value)))
                continue
            pending.append((value, environment, recipe))
            pending.extend((child, env, None) for child, env in reversed(recipe.children))
        result = self._memo[self._key(root, ())]
        if result.variables is not None:
            raise AssertionError("internal binding token escaped the adapter")
        return result


def alpha_hash(expression: Base) -> int:
    """Hash one expression modulo alpha equivalence using a temporary cache.

    Equal fingerprints are only candidate matches: confirm with alpha_equivalent.
    Use AlphaHasher when processing several expressions with shared subgraphs.
    """
    return AlphaHasher().fingerprint(expression)
