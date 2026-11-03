from __future__ import annotations

import weakref
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Concatenate, ParamSpec, TypeVar, cast

from .names import is_renamed_label, renamed_label
from .nodes import AbstractExpression, AbstractFunctionSpace, Base, Function, Index
from .types import Construction

if TYPE_CHECKING:
    from .nodes.indices import AbstractExtent


T = TypeVar("T", bound=Base)
P = ParamSpec("P")
K = TypeVar("K", bound=Base)
R = TypeVar("R")


def operands(e: Base, names: Sequence[str] | None = None) -> list[Base]:
    # The nodes on the edges of e, in order, through tuples, or those of the named arguments.
    return list(edges(e) if names is None else nodes_in(e, names))


def edges(e: Base) -> tuple[Base, ...]:
    # All construction dependencies, including spaces, maps, labels and literal parts.
    memo = vars(e)
    if "edges" not in memo:
        memo["edges"] = nodes_in(e, e.argument_names)
    return cast("tuple[Base, ...]", memo["edges"])


def nodes_in(e: Base, names: Sequence[str]) -> tuple[Base, ...]:
    out: list[Base] = []

    def walk(x: Construction) -> None:
        if isinstance(x, Base):
            out.append(x)
        elif isinstance(x, tuple):
            for y in x:
                walk(y)

    for name in names:
        walk(getattr(e, name))
    return tuple(out)


def rebuild(e: Base, **changes: Construction) -> Construction:
    # An unknown name adds an argument, which the class rejects.
    values = dict(zip(e.argument_names, e.construction_arguments)) | changes
    from .nodes import ConjugateSpace, DualSpace
    from .spaces import conjugate_space, dual_space

    if isinstance(e, ConjugateSpace):
        return conjugate_space(cast("AbstractFunctionSpace", values["space"]))
    if isinstance(e, DualSpace):
        return dual_space(
            cast("AbstractFunctionSpace | tuple[AbstractFunctionSpace, ...]", values["space"])
        )
    return type(e)(*values.values())


def expressions(e: Base) -> list[AbstractExpression]:
    # The operands of e that are expressions, which a grammar restricts.
    return [x for x in nodes_in(e, lowered(e)) if isinstance(x, AbstractExpression)]


def map_fields(
    e: Base, fn: Callable[[Base], Construction], names: Sequence[str] | None = None
) -> Construction:
    # Rebuild e with fn applied to every node in the named fields.
    def walk(x: Construction) -> Construction:
        if isinstance(x, Base):
            return fn(x)
        if isinstance(x, tuple):
            return tuple(walk(y) for y in x)
        return x

    names = e.argument_names if names is None else names
    return rebuild(e, **{n: walk(getattr(e, n)) for n in names})


def occurs(r: Base, e: Construction) -> bool:
    # Whether r is free in e, kept on e and weakly in r so that each is freed with its node.
    if not isinstance(e, Base):
        return False
    memo = cast(
        "weakref.WeakKeyDictionary[Base, bool]",
        vars(e).setdefault("occurrences", weakref.WeakKeyDictionary()),
    )
    if r not in memo:
        memo[r] = e == r or any(occurs(r, x) for x in operands(e, scoped(r, e)))
    return memo[r]


def scoped(r: Base, e: Base) -> list[str]:
    # The arguments of e in which r can be free, without the scope if r names a bound name.
    names = list(e.argument_names) if isinstance(r, Index) else list(lowered(e))
    names = [n for n in names if n not in e.labels]
    if any(occurs(n, r) for n in e.bound()):
        names = [n for n in names if n not in e.scope]
    return names


def rename(node: Base, name: Index | Function, avoid: Sequence[Construction]) -> Index | Function:
    # The label refers to the bound operands and to avoid, never to the binding node.
    label = renamed_label(node, name, avoid)
    if isinstance(name, Index):
        return type(name)(label)
    return Function(name.space, label)


def freshen(e: T, avoid: Sequence[Construction]) -> T:
    # Rename the bound names of e that are free in an expression of avoid.
    renames: dict[Base, Construction] = {
        n: rename(e, n, avoid) for n in e.bound() if any(occurs(n, s) for s in avoid)
    }
    return cast("T", map_fields(e, lambda x: substitute(x, renames), e.scope)) if renames else e


def substitute(e: Base, mapping: Mapping[K, Construction]) -> Construction:
    # Simultaneous capture-avoiding substitution of mapping[r] for each r in e.
    return memoized(substituted)(e, tuple(sorted(mapping.items(), key=lambda item: item[0].digest)))


def substituted(
    walk: Callable[[Base, tuple[tuple[Base, Construction], ...]], Construction],
    e: Base,
    items: tuple[tuple[Base, Construction], ...],
    dependencies: bool = True,
) -> Construction:
    mapping = dict(items)
    if e in mapping:
        return mapping[e]
    from .nodes import GatheredIndex, Zero
    from .validation import merge_indices

    if isinstance(e, Zero):
        # Zero's free-index signature changes when an index is selected or gathered.
        pairs: list[tuple[tuple[Index, int | AbstractExtent], ...]] = []
        for i, extent in e.free_indices:
            entry = walk(i, items)
            size = (
                cast("int | AbstractExtent", walk(extent, items))
                if isinstance(extent, Base)
                else extent
            )
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
        new = map_fields(
            e, lambda x: walk(x, items), ("value_shape", "argument_spaces", "scalar_type")
        )
        return rebuild(cast("Base", new), free_indices=merge_indices(*pairs))
    if e.labels:
        if not dependencies:
            return e
        # A derived name with a replaced object in its label is rebuilt, and a renamed name is not.
        if is_renamed_label(getattr(e, "label")):
            return map_fields(
                e, lambda x: walk(x, items), [n for n in e.argument_names if n not in e.labels]
            )
        return map_fields(
            e,
            lambda x: walk(x, items),
            e.argument_names,
        )
    if not e.scope:
        return map_fields(
            e,
            lambda x: walk(x, items),
            e.argument_names if dependencies else (*e.value_fields, *e.index_fields),
        )
    inner = {r: s for r, s in mapping.items() if not any(occurs(n, r) for n in e.bound())}
    e = freshen(e, list(inner.values()))
    outside = [
        n
        for n in (e.argument_names if dependencies else (*e.value_fields, *e.index_fields))
        if n not in e.scope
    ]
    e = cast("Base", map_fields(e, lambda x: walk(x, items), outside))
    return map_fields(e, lambda x: walk(x, tuple(inner.items())), e.scope)


def memoized(rule: Callable[Concatenate[Callable[P, R], P], R]) -> Callable[P, R]:
    # The recursion of rule through one memo, which lasts as long as the returned function.
    memo: dict[tuple[object, ...], R] = {}

    def recurse(*args: P.args, **kwargs: P.kwargs) -> R:
        key = (*args, *kwargs.items())
        if key not in memo:
            memo[key] = rule(recurse, *args, **kwargs)
        return memo[key]

    return recurse


def lowered(e: Base) -> tuple[str, ...]:
    # Only explicitly declared value operands are evaluated.
    return e.value_fields


def bottom_up(
    f: AbstractExpression, rewrite: Callable[[AbstractExpression], AbstractExpression]
) -> AbstractExpression:
    # Rewrite each node after its operands, once for each node of the graph of f.
    def visit(
        walk: Callable[[AbstractExpression], AbstractExpression], e: AbstractExpression
    ) -> AbstractExpression:
        return rewrite(map_values(e, walk))

    return memoized(visit)(f)


def everywhere(e: AbstractExpression, holds: Callable[[AbstractExpression], bool]) -> bool:
    # Whether holds is true at every node of the graph of e.
    def visit(walk: Callable[[AbstractExpression], bool], n: AbstractExpression) -> bool:
        return holds(n) and all(map(walk, expressions(n)))

    return memoized(visit)(e)


def replace(e: Base, mapping: dict[Base, Construction]) -> Construction:
    """Substitute simultaneously through construction dependencies, respecting binding."""
    if type(mapping) is not dict:
        raise TypeError("replace takes a graph object and a dictionary")
    return substitute(e, mapping)


def substitute_values(
    e: AbstractExpression, mapping: Mapping[K, Construction]
) -> AbstractExpression:
    # Value instantiation holds geometry and identity metadata fixed.
    items = tuple(sorted(mapping.items(), key=lambda item: item[0].digest))

    def visit(
        walk: Callable[[Base, tuple[tuple[Base, Construction], ...]], Construction],
        node: Base,
        pairs: tuple[tuple[Base, Construction], ...],
    ) -> Construction:
        return substituted(walk, node, pairs, False)

    return cast("AbstractExpression", memoized(visit)(e, items))


def map_values(
    e: AbstractExpression,
    fn: Callable[[AbstractExpression], AbstractExpression],
    names: Sequence[str] | None = None,
) -> AbstractExpression:
    # The node declares these fields to contain expressions, including tuple members.
    return cast(
        "AbstractExpression",
        map_fields(
            e,
            lambda child: fn(cast("AbstractExpression", child)),
            e.value_fields if names is None else names,
        ),
    )
