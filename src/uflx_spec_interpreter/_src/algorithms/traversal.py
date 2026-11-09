"""Graph reconstruction and traversal of dependencies or expression values."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Concatenate, ParamSpec, TypeVar, cast

from ..nodes import AbstractExpression, AbstractFunctionSpace, Base
from ..nodes.base import construction_equal
from ..nodes.structure import expression_operands
from ..nodes.types import Construction

_P = ParamSpec("_P")
_R = TypeVar("_R")


def rebuild(e: Base, **changes: Construction) -> Construction:
    # An unknown name adds an argument, which the class rejects.
    values = dict(zip(e.argument_names, e.construction_arguments)) | changes
    if tuple(values) == e.argument_names and construction_equal(
        tuple(values.values()), e.construction_arguments
    ):
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


def memoized(rule: Callable[Concatenate[Callable[_P, _R], _P], _R]) -> Callable[_P, _R]:
    # The recursion of rule through one memo, which lasts as long as the returned function.
    memo: dict[tuple[object, ...], _R] = {}

    def recurse(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        key = (*args, *kwargs.items())
        if key not in memo:
            memo[key] = rule(recurse, *args, **kwargs)
        return memo[key]

    return recurse


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
        return holds(n) and all(map(walk, expression_operands(n)))

    return memoized(visit)(e)


def map_values(
    e: AbstractExpression,
    fn: Callable[[AbstractExpression], AbstractExpression],
    names: Sequence[str] | None = None,
) -> AbstractExpression:
    # The node declares these fields to contain expressions, including tuple members.
    fields = e.value_fields if names is None else names
    mapped = map_fields(e, lambda child: fn(cast("AbstractExpression", child)), fields)
    return cast("AbstractExpression", mapped)
