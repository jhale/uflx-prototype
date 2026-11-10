# SPDX-License-Identifier: MIT
"""Deterministic fingerprints modulo the interpreter's lexical alpha equivalence.

UFLx-specific adapter to the reference algorithm port in _alpha_summary.py,
which contains the paper citation and upstream implementation's license notice.

Summary assembly and cache keys are pure; the adapter owns graph caches and
fresh-token allocation. Traversal is iterative to support deep expression DAGs.

Labels retain exact construction identity; parameter signatures are outside their
own binder. Dependency sets share unchanged branches. Context projection still
scans enclosing bindings: no whole-adapter complexity bound is claimed.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import NamedTuple, TypeAlias

from ..nodes import Base, Function, Index
from ..nodes.base import construction_encoding
from ..nodes.types import Construction
from ._alpha_dependencies import Dependency, contains, singleton, union, without
from ._alpha_summary import Summary, bind, constructor, variable
from .binding import binding_signature, construction_nodes

_Environment: TypeAlias = tuple[tuple[Function | Index, int], ...]
_Key: TypeAlias = tuple[object, ...]


@dataclass(frozen=True, eq=False, slots=True)
class _Identity:
    """An identity key that keeps its tuple alive without a separate retention map."""

    value: tuple[Construction, ...]

    def __hash__(self) -> int:
        return id(self.value)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _Identity) and self.value is other.value


class _NodeInfo(NamedTuple):
    signature: tuple[Construction, ...]
    dependencies: Dependency | None


class _Request(NamedTuple):
    value: Construction
    environment: _Environment
    key: _Key


class _Recipe(NamedTuple):
    tag: object
    children: tuple[_Request, ...]
    tokens: tuple[int, ...] | None
    scoped: tuple[bool, ...]


def _key(value: Construction, environment: _Environment) -> _Key:
    """Build a side-effect-free key from an already projected environment."""
    if isinstance(value, Base):
        return ("node", value, environment)
    if isinstance(value, tuple):
        # Ordinary tuple equality would conflate 1 and 1.0.
        return ("tuple", _Identity(value), environment)
    return ("literal", construction_encoding(value))


def _value_dependencies(value: Construction, info: Mapping[Base, _NodeInfo]) -> Dependency | None:
    result = None
    for node in construction_nodes(value):
        result = union(result, info[node].dependencies)
    return result


def _assemble(recipe: _Recipe, results: tuple[Summary, ...]) -> Summary:
    """Combine child summaries independently of traversal, caches and name allocation."""
    if recipe.tokens is None:
        return constructor(recipe.tag, results)
    signature, *fields = results
    inside = constructor(
        ("scoped",),
        tuple(result for scoped, result in zip(recipe.scoped, fields, strict=True) if scoped),
    )
    outside = tuple(
        result for scoped, result in zip(recipe.scoped, fields, strict=True) if not scoped
    )
    return constructor(recipe.tag, (signature, bind(recipe.tokens, inside), *outside))


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
        self._info: dict[Base, _NodeInfo] = {}
        self._names: dict[Function | Index, int] = {}
        self._memo: dict[_Key, Summary] = {}
        self._tokens = 0

    def clear(self) -> None:
        """Release cached graphs and summaries; the hasher can then be reused."""
        self._info.clear()
        self._names.clear()
        self._memo.clear()
        self._tokens = 0

    def fingerprint(self, expression: Base) -> int:
        """Return a candidate key invariant under lexical renaming of bound names."""
        return self._summary(expression).fingerprint

    def _prepare(self, root: Base) -> None:
        pending: list[tuple[Base, tuple[Construction, ...] | None]] = [(root, None)]
        while pending:
            node, signature = pending.pop()
            if node in self._info:
                continue
            fields = tuple(f for f in node.argument_names if f not in node.labels)
            if signature is None:
                signature = binding_signature(node)
                pending.append((node, signature))
                children = list(construction_nodes(signature))
                for field in fields:
                    children.extend(construction_nodes(getattr(node, field)))
                pending.extend((child, None) for child in children if child not in self._info)
                continue
            deps = _value_dependencies(signature, self._info)
            if isinstance(node, Function | Index):
                key = self._names.setdefault(node, len(self._names))
                deps = union(deps, singleton(key))
            keys = tuple(self._names[name] for name in node.bound())
            for field in fields:
                child = _value_dependencies(getattr(node, field), self._info)
                deps = union(deps, without(child, keys) if field in node.scope else child)
            self._info[node] = _NodeInfo(signature, deps)

    def _request(self, value: Construction, environment: _Environment) -> _Request:
        if isinstance(value, Base):
            deps = self._info[value].dependencies
            environment = tuple(
                pair for pair in environment if contains(deps, self._names[pair[0]])
            )
        return _Request(value, environment, _key(value, environment))

    def _recipe(self, value: Base | tuple[Construction, ...], env: _Environment) -> _Recipe:
        if isinstance(value, tuple):
            return _Recipe(
                ("tuple",), tuple(self._request(child, env) for child in value), None, ()
            )
        names = value.bound()
        tokens = tuple(range(self._tokens, self._tokens + len(names)))
        self._tokens += len(names)
        inner = tuple(pair for pair in env if pair[0] not in names) + tuple(
            zip(names, tokens, strict=True)
        )
        children = [self._request(self._info[value].signature, env)]
        scoped = tuple(field in value.scope for field in value.argument_names)
        for field, in_scope in zip(value.argument_names, scoped, strict=True):
            child: Construction = getattr(value, field)
            if field in value.labels:
                child = ("opaque", construction_encoding(child))
            children.append(self._request(child, inner if in_scope else env))
        return _Recipe(
            ("node", value.identifier, value.argument_names, value.scope),
            tuple(children),
            tokens,
            scoped,
        )

    def _summary(self, root: Base) -> Summary:
        self._prepare(root)
        request = self._request(root, ())
        pending: list[tuple[_Request, _Recipe | None]] = [(request, None)]
        while pending:
            current, recipe = pending.pop()
            if current.key in self._memo:
                continue
            value, environment, key = current
            if recipe is not None:
                self._memo[key] = _assemble(
                    recipe, tuple(self._memo[c.key] for c in recipe.children)
                )
                continue
            if isinstance(value, Function | Index):
                token = next((token for name, token in environment if name == value), None)
                if token is not None:
                    self._memo[key] = constructor(
                        ("bound-type", value.identifier), (variable(token),)
                    )
                    continue
            if isinstance(value, Base | tuple):
                recipe = self._recipe(value, environment)
                pending.append((current, recipe))
                pending.extend((child, None) for child in reversed(recipe.children))
            else:
                self._memo[key] = constructor(("literal", construction_encoding(value)))
        result = self._memo[request.key]
        if result.variables is not None:
            raise AssertionError("internal binding token escaped the adapter")
        return result


def alpha_hash(expression: Base) -> int:
    """Hash one expression modulo alpha equivalence using a temporary cache.

    Equal fingerprints are only candidate matches: confirm with alpha_equivalent.
    Use AlphaHasher when processing several expressions with shared subgraphs.
    """
    return AlphaHasher().fingerprint(expression)
