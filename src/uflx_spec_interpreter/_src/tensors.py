from itertools import combinations

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


def tensor(entries):
    return Tensor(entries)


def indexed(expression, indices):
    if indices == () and isinstance(expression, AbstractExpression) and expression.value_shape:
        raise IndexError("full indexing must cover every axis")
    return empty_indices(expression, indices) if indices == () else Indexed(expression, indices)


def component_tensor(expression, indices):
    return (
        empty_indices(expression, indices)
        if indices == ()
        else ComponentTensor(expression, indices)
    )


def index_sum(expression, indices):
    return empty_indices(expression, indices) if indices == () else IndexSum(expression, indices)


def inverse(A):
    return Inverse(A)


def identity(n):
    if n == 0:
        return Zero((0, 0), (), (), ScalarType.real)
    return Tensor(
        tuple(Tensor(tuple(Integer(1 if r == c else 0) for c in range(n))) for r in range(n))
    )


def permutation_tensor(n):
    # The Levi-Civita symbol of rank n, the sign of each permutation and 0 elsewhere.
    def entries(prefix):
        if len(prefix) == n:
            return Integer(sign(prefix) if len(set(prefix)) == n else 0)
        return Tensor(tuple(entries((*prefix, k)) for k in range(n)))

    return entries(())


def transpose(A):
    k, l = derived_indices("uflx.transpose", (A,), 2)
    return ComponentTensor(Indexed(A, (k, l)), (l, k))


def matrix_product(first, second):
    i, k, j = derived_indices("uflx.matrix_product", (first, second), 3)
    a = (i,) if rank(first) == 2 else ()
    b = (j,) if rank(second) == 2 else ()
    contracted = IndexSum(Product(Indexed(first, (*a, k)), Indexed(second, (k, *b))), (k,))
    return ComponentTensor(contracted, a + b) if a + b else contracted


def component(f, entries):
    if not isinstance(f, AbstractExpression) or type(entries) is not tuple:
        raise TypeError("component takes an expression and a tuple")
    if not entries:
        return f
    if len(entries) < rank(f):
        K = derived_indices("uflx.component", (f, entries), rank(f) - len(entries))
        return ComponentTensor(Indexed(f, (*entries, *K)), K)
    return Indexed(f, entries)


def norm(u):
    return SquareRoot(matrix_product(u, u))


def unit(u):
    return divide(u, norm(u))


def determinant(A):
    return Determinant(A)


def sign(p):
    return (-1) ** sum(p[a] > p[b] for a, b in combinations(range(len(p)), 2))


def empty_indices(expression, indices):
    if not isinstance(expression, AbstractExpression) or type(indices) is not tuple:
        raise TypeError("index notation takes an expression and a tuple")
    if expression.value_shape:
        raise ValueError("empty indices require a scalar")
    return expression
