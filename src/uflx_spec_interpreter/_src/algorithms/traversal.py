"""Graph reconstruction and traversal of dependencies or expression values."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Concatenate, ParamSpec, TypeVar, cast

from ..nodes import AbstractExpression, AbstractFunctionSpace, Base
from ..nodes.types import _Construction

_P = ParamSpec("_P")
_R = TypeVar("_R")
__all__ = [
    "_bottom_up",
    "_edges",
    "_everywhere",
    "_expressions",
    "_map_fields",
    "_map_values",
    "_memoized",
    "_nodes_in",
    "_operands",
    "_rebuild",
]


def _operands(e: Base, names: Sequence[str] | None = None) -> list[Base]:
    # The nodes on the edges of e, in order, through tuples, or those of the named arguments.
    return list(_edges(e) if names is None else _nodes_in(e, names))


def _edges(e: Base) -> tuple[Base, ...]:
    # All construction dependencies, including spaces, maps, labels and literal parts.
    memo = vars(e)
    if "edges" not in memo:
        memo["edges"] = _nodes_in(e, e.argument_names)
    return cast("tuple[Base, ...]", memo["edges"])


def _nodes_in(e: Base, names: Sequence[str]) -> tuple[Base, ...]:
    out: list[Base] = []

    def walk(x: _Construction) -> None:
        if isinstance(x, Base):
            out.append(x)
        elif isinstance(x, tuple):
            for y in x:
                walk(y)

    for name in names:
        walk(getattr(e, name))
    return tuple(out)


def _rebuild(e: Base, **changes: _Construction) -> _Construction:
    # An unknown name adds an argument, which the class rejects.
    values = dict(zip(e.argument_names, e.construction_arguments)) | changes
    if tuple(values) == e.argument_names and tuple(values.values()) == e.construction_arguments:
        return e
    from ..nodes import ConjugateSpace, DualSpace
    from ..spaces import conjugate_space, dual_space

    if isinstance(e, ConjugateSpace):
        return conjugate_space(cast("AbstractFunctionSpace", values["space"]))
    if isinstance(e, DualSpace):
        return dual_space(
            cast("AbstractFunctionSpace | tuple[AbstractFunctionSpace, ...]", values["space"])
        )
    return type(e)(*values.values())


def _expressions(e: Base) -> list[AbstractExpression]:
    # The operands of e that are expressions, which a grammar restricts.
    return [x for x in _nodes_in(e, e.value_fields) if isinstance(x, AbstractExpression)]


def _map_fields(
    e: Base, fn: Callable[[Base], _Construction], names: Sequence[str] | None = None
) -> _Construction:
    # Rebuild e with fn applied to every node in the named fields.
    def walk(x: _Construction) -> _Construction:
        if isinstance(x, Base):
            return fn(x)
        if isinstance(x, tuple):
            return tuple(walk(y) for y in x)
        return x

    names = e.argument_names if names is None else names
    return _rebuild(e, **{n: walk(getattr(e, n)) for n in names})


def _memoized(rule: Callable[Concatenate[Callable[_P, _R], _P], _R]) -> Callable[_P, _R]:
    # The recursion of rule through one memo, which lasts as long as the returned function.
    memo: dict[tuple[object, ...], _R] = {}

    def recurse(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        key = (*args, *kwargs.items())
        if key not in memo:
            memo[key] = rule(recurse, *args, **kwargs)
        return memo[key]

    return recurse


def _bottom_up(
    f: AbstractExpression, rewrite: Callable[[AbstractExpression], AbstractExpression]
) -> AbstractExpression:
    # Rewrite each node after its operands, once for each node of the graph of f.
    def visit(
        walk: Callable[[AbstractExpression], AbstractExpression], e: AbstractExpression
    ) -> AbstractExpression:
        return rewrite(_map_values(e, walk))

    return _memoized(visit)(f)


def _everywhere(e: AbstractExpression, holds: Callable[[AbstractExpression], bool]) -> bool:
    # Whether holds is true at every node of the graph of e.
    def visit(walk: Callable[[AbstractExpression], bool], n: AbstractExpression) -> bool:
        return holds(n) and all(map(walk, _expressions(n)))

    return _memoized(visit)(e)


def _map_values(
    e: AbstractExpression,
    fn: Callable[[AbstractExpression], AbstractExpression],
    names: Sequence[str] | None = None,
) -> AbstractExpression:
    # The node declares these fields to contain expressions, including tuple members.
    fields = e.value_fields if names is None else names
    mapped = _map_fields(e, lambda child: fn(cast("AbstractExpression", child)), fields)
    return cast("AbstractExpression", mapped)
