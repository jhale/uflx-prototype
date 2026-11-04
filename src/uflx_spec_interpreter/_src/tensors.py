from __future__ import annotations

from itertools import combinations
from typing import TYPE_CHECKING

from .algorithms.names import _derived_indices
from .arithmetic import divide, multiply
from .nodes import (
    AbstractExpression,
    BroadcastTo,
    ComponentTensor,
    Determinant,
    Indexed,
    IndexSum,
    Integer,
    Inverse,
    ScalarType,
    SquareRoot,
    Tensor,
    Zero,
)

if TYPE_CHECKING:
    from .nodes.indices import AbstractExtent, GatheredIndex, Index


def broadcast_to(
    value: AbstractExpression, value_shape: tuple[int | AbstractExtent, ...]
) -> BroadcastTo:
    return BroadcastTo(value, value_shape)


def tensor(entries: tuple[AbstractExpression, ...]) -> Tensor:
    return Tensor(entries)


def indexed(
    expression: AbstractExpression, indices: tuple[int | Index | GatheredIndex, ...]
) -> AbstractExpression:
    if indices == () and expression.value_shape:
        raise IndexError("full indexing must cover every axis")
    return _empty_indices(expression) if indices == () else Indexed(expression, indices)


def component_tensor(
    expression: AbstractExpression, indices: tuple[Index, ...]
) -> AbstractExpression:
    return _empty_indices(expression) if indices == () else ComponentTensor(expression, indices)


def index_sum(expression: AbstractExpression, indices: tuple[Index, ...]) -> AbstractExpression:
    return _empty_indices(expression) if indices == () else IndexSum(expression, indices)


def inverse(A: AbstractExpression) -> AbstractExpression:
    return Inverse(A)


def _identity(n: int) -> AbstractExpression:
    if n == 0:
        return Zero((0, 0), (), (), ScalarType.real)
    rows: list[Tensor] = []
    for row in range(n):
        entries = tuple(Integer(1 if row == column else 0) for column in range(n))
        rows.append(Tensor(entries))
    return Tensor(tuple(rows))


def _permutation_tensor(n: int) -> AbstractExpression:
    # The Levi-Civita symbol of rank n, the sign of each permutation and 0 elsewhere.
    def entries(prefix: tuple[int, ...]) -> AbstractExpression:
        if len(prefix) == n:
            return Integer(_sign(prefix) if len(set(prefix)) == n else 0)
        return Tensor(tuple(entries((*prefix, k)) for k in range(n)))

    return entries(())


def transpose(A: AbstractExpression) -> AbstractExpression:
    k, l = _derived_indices("uflx.transpose", (A,), 2)
    return ComponentTensor(Indexed(A, (k, l)), (l, k))


def matrix_product(first: AbstractExpression, second: AbstractExpression) -> AbstractExpression:
    i, k, j = _derived_indices("uflx.matrix_product", (first, second), 3)
    a = (i,) if len(first.value_shape) == 2 else ()
    b = (j,) if len(second.value_shape) == 2 else ()
    contracted = IndexSum(multiply(Indexed(first, (*a, k)), Indexed(second, (k, *b))), (k,))
    return ComponentTensor(contracted, a + b) if a + b else contracted


def component(
    f: AbstractExpression, entries: tuple[int | Index | GatheredIndex, ...]
) -> AbstractExpression:
    if not entries:
        return f
    if len(entries) < len(f.value_shape):
        K = _derived_indices("uflx.component", (f, entries), len(f.value_shape) - len(entries))
        return ComponentTensor(Indexed(f, (*entries, *K)), K)
    return Indexed(f, entries)


def _norm(u: AbstractExpression) -> AbstractExpression:
    return SquareRoot(matrix_product(u, u))


def _unit(u: AbstractExpression) -> AbstractExpression:
    return divide(u, _norm(u))


def determinant(A: AbstractExpression) -> AbstractExpression:
    return Determinant(A)


def _sign(p: tuple[int, ...]) -> int:
    pairs = combinations(range(len(p)), 2)
    inversions = sum(p[first] > p[second] for first, second in pairs)
    return -1 if inversions % 2 else 1


def _empty_indices(expression: AbstractExpression) -> AbstractExpression:
    if expression.value_shape:
        raise ValueError("empty indices require a scalar")
    return expression
