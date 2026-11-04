from __future__ import annotations

import weakref
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, TypeVar, cast

from .names import _is_renamed_label, _renamed_label
from .nodes import AbstractExpression, Base, Function, Index, Zero
from .traversal import (
    _map_fields,
    _memoized,
    _operands,
    _rebuild,
)
from .types import _Construction

if TYPE_CHECKING:
    from .nodes.indices import AbstractExtent


_T = TypeVar("_T", bound=Base)
_K = TypeVar("_K", bound=Base)


# Shared implementation helpers; the package public API is declared in __init__.py.
__all__ = ["_freshen", "_occurs", "_substitute", "_substitute_values", "replace"]


def _occurs(target: Base, node: _Construction) -> bool:
    # Cache free occurrence on the expression, holding the target only weakly.
    if not isinstance(node, Base):
        return False
    memo = cast(
        "weakref.WeakKeyDictionary[Base, bool]",
        vars(node).setdefault("occurrences", weakref.WeakKeyDictionary()),
    )
    if target not in memo:
        memo[target] = node == target or any(
            _occurs(target, child) for child in _operands(node, _scoped(target, node))
        )
    return memo[target]


def _scoped(target: Base, node: Base) -> list[str]:
    # A bound name cannot occur free inside its own scope. Labels are identities.
    fields = list(node.argument_names) if isinstance(target, Index) else list(node.value_fields)
    fields = [name for name in fields if name not in node.labels]
    if _bound_in(target, node):
        fields = [name for name in fields if name not in node.scope]
    return fields


def _bound_in(target: Base, node: Base) -> bool:
    """Whether entering this binder would capture a free name of the target."""
    return any(_occurs(name, target) for name in node.bound())


def _rename(node: Base, name: Index | Function, avoid: Sequence[_Construction]) -> Index | Function:
    # The label refers to the bound operands and to avoid, never to the binding node.
    label = _renamed_label(node, name, avoid)
    if isinstance(name, Index):
        return type(name)(label)
    return Function(name.space, label)


def _freshen(node: _T, avoid: Sequence[_Construction]) -> _T:
    # Freshen only names that could capture something in the inserted values.
    renames: dict[Base, _Construction] = {
        name: _rename(node, name, avoid)
        for name in node.bound()
        if any(_occurs(name, replacement) for replacement in avoid)
    }
    if not renames:
        return node
    return cast("_T", _map_fields(node, lambda child: _substitute(child, renames), node.scope))


def _substitute(node: Base, mapping: Mapping[_K, _Construction]) -> _Construction:
    # Sort once so dictionary insertion order cannot affect fresh names.
    ordered = sorted(mapping.items(), key=lambda pair: pair[0].digest)
    return _memoized(_substituted)(node, tuple(ordered))


def _substitute_zero(node: Zero, replace: Callable[[Base], _Construction]) -> _Construction:
    from .nodes import GatheredIndex
    from .validation import _entries, _merge_indices

    # Zero's free-index signature changes when an index is selected or gathered.
    pairs: list[tuple[tuple[Index, int | AbstractExtent], ...]] = []
    for i, extent in node.free_indices:
        entry = replace(i)
        size = cast("int | AbstractExtent", replace(extent)) if isinstance(extent, Base) else extent
        if not (type(entry) is int or isinstance(entry, Index | GatheredIndex)):
            raise TypeError("replacement of an index must be an index entry")
        pairs.append(_entries((size,), (entry,)))
    new = _map_fields(node, replace, ("value_shape", "argument_spaces", "scalar_type"))
    return _rebuild(cast("Base", new), free_indices=_merge_indices(*pairs))


def _substituted(
    walk: Callable[[Base, tuple[tuple[Base, _Construction], ...]], _Construction],
    node: Base,
    pairs: tuple[tuple[Base, _Construction], ...],
    dependencies: bool = True,
) -> _Construction:
    mapping = dict(pairs)
    if node in mapping:
        return mapping[node]
    if isinstance(node, Zero):
        return _substitute_zero(node, lambda child: walk(child, pairs))
    if node.labels and not dependencies:
        return node
    fields = node.argument_names if dependencies else (*node.value_fields, *node.index_fields)
    # Derived labels follow substitutions; freshened binder labels remain fixed.
    if _is_renamed_label(getattr(node, "label", None)):
        fields = tuple(name for name in fields if name not in node.labels)
    if node.labels or not node.scope:
        return _map_fields(node, lambda child: walk(child, pairs), fields)
    scoped_pairs = {
        target: replacement for target, replacement in pairs if not _bound_in(target, node)
    }
    node = _freshen(node, list(scoped_pairs.values()))
    unbound_fields = [name for name in fields if name not in node.scope]
    node = cast("Base", _map_fields(node, lambda child: walk(child, pairs), unbound_fields))
    inner_pairs = tuple(scoped_pairs.items())
    return _map_fields(node, lambda child: walk(child, inner_pairs), node.scope)


def replace(e: Base, mapping: dict[Base, _Construction]) -> _Construction:
    """Substitute simultaneously through construction dependencies, respecting binding."""
    return _substitute(e, mapping)


def _substitute_values(
    node: AbstractExpression, mapping: Mapping[_K, _Construction]
) -> AbstractExpression:
    # Value instantiation holds geometry and identity metadata fixed.
    pairs = tuple(sorted(mapping.items(), key=lambda item: item[0].digest))

    def visit(
        walk: Callable[[Base, tuple[tuple[Base, _Construction], ...]], _Construction],
        node: Base,
        pairs: tuple[tuple[Base, _Construction], ...],
    ) -> _Construction:
        return _substituted(walk, node, pairs, False)

    return cast("AbstractExpression", _memoized(visit)(node, pairs))
