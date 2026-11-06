from __future__ import annotations

import weakref
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, TypeVar, cast

from ..nodes import AbstractExpression, Base, Function, Index, Zero
from ..nodes.structure import operands
from ..nodes.types import Construction
from .binding import binding_signature, scoped_body_fields
from .traversal import map_fields, memoized, rebuild

if TYPE_CHECKING:
    from ..nodes.indices import AbstractExtent


_T = TypeVar("_T", bound=Base)
_K = TypeVar("_K", bound=Base)


def occurs(target: Base, node: Construction) -> bool:
    """Test free occurrence through construction dependencies, excluding labels."""
    return _occurs(target, node, True)


def occurs_in_values(target: Base, node: Construction) -> bool:
    """Test free occurrence in values, holding domains and spaces fixed."""
    return _occurs(target, node, False)


def occurs_in_metadata(target: Base, node: Construction) -> bool:
    """Detect dependencies outside value/index operands, including nested metadata."""

    def visit(node: Construction) -> bool:
        if isinstance(node, tuple):
            return any(visit(item) for item in node)
        if not isinstance(node, Base):
            return False
        if node in memo:
            return memo[node]
        result = occurs(target, binding_signature(node))
        for field in _scoped(target, node, True):
            operand = getattr(node, field)
            result |= (
                visit(operand)
                if field in (*node.value_fields, *node.index_fields)
                else occurs(target, operand)
            )
        memo[node] = result
        return result

    memo: dict[Base, bool] = {}
    return visit(node)


def _occurs(target: Base, node: Construction, dependencies: bool) -> bool:
    if isinstance(node, tuple):
        return any(_occurs(target, item, dependencies) for item in node)
    if not isinstance(node, Base):
        return False
    cache_name = "occurrences" if dependencies else "value_occurrences"
    memo = cast(
        "weakref.WeakKeyDictionary[Base, bool]",
        vars(node).setdefault(cache_name, weakref.WeakKeyDictionary()),
    )
    if target not in memo:
        declaration_dependency = dependencies and _occurs(target, binding_signature(node), True)
        memo[target] = (
            node == target
            or declaration_dependency
            or any(
                _occurs(target, child, dependencies)
                for child in operands(node, _scoped(target, node, dependencies))
            )
        )
    return memo[target]


def _scoped(target: Base, node: Base, dependencies: bool) -> list[str]:
    fields = node.argument_names if dependencies else (*node.value_fields, *node.index_fields)
    fields = [name for name in fields if name not in (*node.labels, *node.binding_fields)]
    if _bound_in(target, node, dependencies):
        fields = [name for name in fields if name not in node.scope]
    return fields


def _bound_in(target: Base, node: Base, dependencies: bool = True) -> bool:
    """Whether the target contains a name bound on entering this scope."""
    return any(_occurs(name, target, dependencies) for name in node.bound())


def _rename(node: Base, name: Index | Function, avoid: Sequence[Construction]) -> Index | Function:
    # The label refers to the bound operands and to avoid, never to the binding node.
    label = _renamed_label(node, name, avoid)
    if isinstance(name, Index):
        return type(name)(label)
    return Function(name.space, label)


def freshen(node: _T, avoid: Sequence[Construction]) -> _T:
    # Freshen only names that could capture something in the inserted values.
    renames: dict[Base, Construction] = {
        name: _rename(node, name, avoid)
        for name in node.bound()
        if any(occurs(name, replacement) for replacement in avoid)
    }
    if not renames:
        return node
    return _rename_scope(node, renames)


def _rename_scope(node: _T, renames: Mapping[_K, Construction]) -> _T:
    return cast("_T", map_fields(node, lambda child: substitute(child, renames), node.scope))


def rename_bound(node: _T, renames: Mapping[Index | Function, Index | Function]) -> _T:
    """Rename declarations simultaneously, preserving type and lexical scope.

    Public adapters validate entry grouping and conjugation. This operation
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


def substitute(node: Base, mapping: Mapping[_K, Construction]) -> Construction:
    # Sort once so dictionary insertion order cannot affect fresh names.
    ordered = sorted(mapping.items(), key=lambda pair: pair[0].digest)
    return memoized(_substituted)(node, tuple(ordered))


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


def _substituted(
    walk: Callable[[Base, tuple[tuple[Base, Construction], ...]], Construction],
    node: Base,
    pairs: tuple[tuple[Base, Construction], ...],
    dependencies: bool = True,
) -> Construction:
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
        return map_fields(node, lambda child: walk(child, pairs), fields)
    scoped_pairs = {
        target: replacement for target, replacement in pairs if not _bound_in(target, node)
    }
    node = freshen(node, list(scoped_pairs.values()))
    unbound_fields = [name for name in fields if name not in node.scope]
    node = cast("Base", map_fields(node, lambda child: walk(child, pairs), unbound_fields))
    inner_pairs = tuple(scoped_pairs.items())
    return map_fields(node, lambda child: walk(child, inner_pairs), node.scope)


def replace(e: Base, mapping: dict[Base, Construction]) -> Construction:
    """Substitute simultaneously through construction dependencies, respecting binding."""
    return substitute(e, mapping)


def substitute_values(
    node: AbstractExpression, mapping: Mapping[_K, Construction]
) -> AbstractExpression:
    # Value instantiation holds geometry and identity metadata fixed.
    pairs = tuple(sorted(mapping.items(), key=lambda item: item[0].digest))

    def visit(
        walk: Callable[[Base, tuple[tuple[Base, Construction], ...]], Construction],
        node: Base,
        pairs: tuple[tuple[Base, Construction], ...],
    ) -> Construction:
        return _substituted(walk, node, pairs, False)

    return cast("AbstractExpression", memoized(visit)(node, pairs))


def _renamed_label(
    node: Base, name: Base, avoid: Sequence[Construction]
) -> tuple[Construction, ...]:
    """Record a capture-avoidance identity without referring to its binding node.

    The scoped operands, original name and ordered avoidance context distinguish
    this renaming. Substitution must subsequently treat the label as fixed.
    """
    body = tuple(getattr(node, field) for field in node.scope)
    return ("uflx.rename", name, body, tuple(avoid))


def _is_renamed_label(label: object) -> bool:
    """Whether a label records a fixed capture-avoidance identity."""
    return isinstance(label, tuple) and label[:1] == ("uflx.rename",)
