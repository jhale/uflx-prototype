from __future__ import annotations

from itertools import combinations
from typing import TYPE_CHECKING

from .arithmetic import divide, rank
from .names import derived_indices
from .nodes import (
    AbstractExpression,
    ComponentTensor,
    Determinant,
    Indexed,
    IndexSum,
    Integer,
    Inverse,
    Product,
    ScalarType,
    SquareRoot,
    Tensor,
    Zero,
)

if TYPE_CHECKING:
    from .nodes.indices import GatheredIndex, Index


def tensor(entries: tuple[AbstractExpression, ...]) -> Tensor:
    return Tensor(entries)


def indexed(
    expression: AbstractExpression, indices: tuple[int | Index | GatheredIndex, ...]
) -> AbstractExpression:
    if indices == () and expression.value_shape:
        raise IndexError("full indexing must cover every axis")
    return empty_indices(expression, indices) if indices == () else Indexed(expression, indices)


def component_tensor(
    expression: AbstractExpression, indices: tuple[Index, ...]
) -> AbstractExpression:
    return (
        empty_indices(expression, indices)
        if indices == ()
        else ComponentTensor(expression, indices)
    )


def index_sum(expression: AbstractExpression, indices: tuple[Index, ...]) -> AbstractExpression:
    return empty_indices(expression, indices) if indices == () else IndexSum(expression, indices)


def inverse(A: AbstractExpression) -> AbstractExpression:
    return Inverse(A)


def identity(n: int) -> AbstractExpression:
    if n == 0:
        return Zero((0, 0), (), (), ScalarType.real)
    return Tensor(
        tuple(Tensor(tuple(Integer(1 if r == c else 0) for c in range(n))) for r in range(n))
    )


def permutation_tensor(n: int) -> AbstractExpression:
    # The Levi-Civita symbol of rank n, the sign of each permutation and 0 elsewhere.
    def entries(prefix: tuple[int, ...]) -> AbstractExpression:
        if len(prefix) == n:
            return Integer(sign(prefix) if len(set(prefix)) == n else 0)
        return Tensor(tuple(entries((*prefix, k)) for k in range(n)))

    return entries(())


def transpose(A: AbstractExpression) -> AbstractExpression:
    k, l = derived_indices("uflx.transpose", (A,), 2)
    return ComponentTensor(Indexed(A, (k, l)), (l, k))


def matrix_product(first: AbstractExpression, second: AbstractExpression) -> AbstractExpression:
    i, k, j = derived_indices("uflx.matrix_product", (first, second), 3)
    a = (i,) if rank(first) == 2 else ()
    b = (j,) if rank(second) == 2 else ()
    contracted = IndexSum(Product(Indexed(first, (*a, k)), Indexed(second, (k, *b))), (k,))
    return ComponentTensor(contracted, a + b) if a + b else contracted


def component(
    f: AbstractExpression, entries: tuple[int | Index | GatheredIndex, ...]
) -> AbstractExpression:
    if type(entries) is not tuple:
        raise TypeError("component takes an expression and a tuple")
    if not entries:
        return f
    if len(entries) < rank(f):
        K = derived_indices("uflx.component", (f, entries), rank(f) - len(entries))
        return ComponentTensor(Indexed(f, (*entries, *K)), K)
    return Indexed(f, entries)


def norm(u: AbstractExpression) -> AbstractExpression:
    return SquareRoot(matrix_product(u, u))


def unit(u: AbstractExpression) -> AbstractExpression:
    return divide(u, norm(u))


def determinant(A: AbstractExpression) -> AbstractExpression:
    return Determinant(A)


def sign(p: tuple[int, ...]) -> int:
    return -1 if sum(p[a] > p[b] for a, b in combinations(range(len(p)), 2)) % 2 else 1


def empty_indices(
    expression: AbstractExpression, indices: tuple[int | Index | GatheredIndex, ...]
) -> AbstractExpression:
    if type(indices) is not tuple:
        raise TypeError("index notation takes an expression and a tuple")
    if expression.value_shape:
        raise ValueError("empty indices require a scalar")
    return expression
