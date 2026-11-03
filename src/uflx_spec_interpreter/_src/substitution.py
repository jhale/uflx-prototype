from __future__ import annotations

import weakref
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, TypeVar, cast

from .names import _is_renamed_label, _renamed_label
from .nodes import AbstractExpression, Base, Function, Index, Zero
from .traversal import (
    _lowered,
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


def _occurs(r: Base, e: _Construction) -> bool:
    # Whether r is free in e, kept on e and weakly in r so that each is freed with its node.
    if not isinstance(e, Base):
        return False
    memo = cast(
        "weakref.WeakKeyDictionary[Base, bool]",
        vars(e).setdefault("occurrences", weakref.WeakKeyDictionary()),
    )
    if r not in memo:
        memo[r] = e == r or any(_occurs(r, x) for x in _operands(e, _scoped(r, e)))
    return memo[r]


def _scoped(r: Base, e: Base) -> list[str]:
    # The arguments of e in which r can be free, without the scope if r names a bound name.
    names = list(e.argument_names) if isinstance(r, Index) else list(_lowered(e))
    names = [n for n in names if n not in e.labels]
    if any(_occurs(n, r) for n in e.bound()):
        names = [n for n in names if n not in e.scope]
    return names


def _rename(node: Base, name: Index | Function, avoid: Sequence[_Construction]) -> Index | Function:
    # The label refers to the bound operands and to avoid, never to the binding node.
    label = _renamed_label(node, name, avoid)
    if isinstance(name, Index):
        return type(name)(label)
    return Function(name.space, label)


def _freshen(e: _T, avoid: Sequence[_Construction]) -> _T:
    # Rename the bound names of e that are free in an expression of avoid.
    renames: dict[Base, _Construction] = {
        n: _rename(e, n, avoid) for n in e.bound() if any(_occurs(n, s) for s in avoid)
    }
    return cast("_T", _map_fields(e, lambda x: _substitute(x, renames), e.scope)) if renames else e


def _substitute(e: Base, mapping: Mapping[_K, _Construction]) -> _Construction:
    # Simultaneous capture-avoiding substitution of mapping[r] for each r in e.
    return _memoized(_substituted)(
        e, tuple(sorted(mapping.items(), key=lambda item: item[0].digest))
    )


def _substitute_zero(e: Zero, replace: Callable[[Base], _Construction]) -> _Construction:
    from .nodes import GatheredIndex
    from .validation import _merge_indices

    # Zero's free-index signature changes when an index is selected or gathered.
    pairs: list[tuple[tuple[Index, int | AbstractExtent], ...]] = []
    for i, extent in e.free_indices:
        entry = replace(i)
        size = cast("int | AbstractExtent", replace(extent)) if isinstance(extent, Base) else extent
        if isinstance(entry, Index):
            pairs.append(((entry, size),))
        elif isinstance(entry, GatheredIndex):
            if entry.extent != size:
                raise IndexError("replacement gather has the wrong extent")
            pairs.append(entry.free_indices)
        elif type(entry) is not int:
            raise TypeError("replacement of an index must be an index entry")
        elif type(size) is not int or not 0 <= entry < size:
            raise IndexError("replacement index is outside the axis")
    new = _map_fields(e, replace, ("value_shape", "argument_spaces", "scalar_type"))
    return _rebuild(cast("Base", new), free_indices=_merge_indices(*pairs))


def _substituted(
    walk: Callable[[Base, tuple[tuple[Base, _Construction], ...]], _Construction],
    e: Base,
    items: tuple[tuple[Base, _Construction], ...],
    dependencies: bool = True,
) -> _Construction:
    mapping = dict(items)
    if e in mapping:
        return mapping[e]
    if isinstance(e, Zero):
        return _substitute_zero(e, lambda x: walk(x, items))
    if e.labels and not dependencies:
        return e
    names = e.argument_names if dependencies else (*e.value_fields, *e.index_fields)
    # Derived labels follow substitutions; freshened binder labels remain fixed.
    if e.labels and _is_renamed_label(getattr(e, "label")):
        names = tuple(n for n in names if n not in e.labels)
    if e.labels or not e.scope:
        return _map_fields(e, lambda x: walk(x, items), names)
    inner = {r: s for r, s in mapping.items() if not any(_occurs(n, r) for n in e.bound())}
    e = _freshen(e, list(inner.values()))
    outside = [n for n in names if n not in e.scope]
    e = cast("Base", _map_fields(e, lambda x: walk(x, items), outside))
    return _map_fields(e, lambda x: walk(x, tuple(inner.items())), e.scope)


def replace(e: Base, mapping: dict[Base, _Construction]) -> _Construction:
    """Substitute simultaneously through construction dependencies, respecting binding."""
    if type(mapping) is not dict:
        raise TypeError("replace takes a graph object and a dictionary")
    return _substitute(e, mapping)


def _substitute_values(
    e: AbstractExpression, mapping: Mapping[_K, _Construction]
) -> AbstractExpression:
    # Value instantiation holds geometry and identity metadata fixed.
    items = tuple(sorted(mapping.items(), key=lambda item: item[0].digest))

    def visit(
        walk: Callable[[Base, tuple[tuple[Base, _Construction], ...]], _Construction],
        node: Base,
        pairs: tuple[tuple[Base, _Construction], ...],
    ) -> _Construction:
        return _substituted(walk, node, pairs, False)

    return cast("AbstractExpression", _memoized(visit)(e, items))
