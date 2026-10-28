from functools import reduce
from itertools import combinations, permutations

from .arithmetic import derived, divide, rank, total
from .nodes import (
    ComponentTensor,
    EllipsisIndex,
    Indexed,
    IndexSum,
    Integer,
    Inverse,
    Product,
    SquareRoot,
    Tensor,
)


def tensor(entries):
    return Tensor(tuple(entries))


def indexed(expression, indices):
    return Indexed(expression, tuple(indices))


def component_tensor(expression, indices):
    return ComponentTensor(expression, tuple(indices))


def index_sum(expression, indices):
    return IndexSum(expression, tuple(indices))


def inverse(A):
    return Inverse(A)


def identity(n):
    return Tensor(
        tuple(Tensor(tuple(Integer(1 if r == c else 0) for c in range(n))) for r in range(n))
    )


def transpose(A):
    k, l = derived("uflx.transpose", (A,), 2)
    return ComponentTensor(Indexed(A, (k, l)), (l, k))


def matrix_product(first, second):
    i, k, j = derived("uflx.matrix_product", (first, second), 3)
    a = (i,) if rank(first) == 2 else ()
    b = (j,) if rank(second) == 2 else ()
    contracted = IndexSum(Product(Indexed(first, (*a, k)), Indexed(second, (k, *b))), (k,))
    return ComponentTensor(contracted, a + b) if a + b else contracted


def component(f, entries):
    if len(entries) < rank(f) and not any(isinstance(e, EllipsisIndex) for e in entries):
        K = derived("uflx.component", (f, entries), rank(f) - len(entries))
        return ComponentTensor(Indexed(f, (*entries, *K)), K)
    return Indexed(f, entries)


def norm(u):
    return SquareRoot(matrix_product(u, u))


def unit(u):
    return divide(u, norm(u))


def determinant(A):
    # The Leibniz formula, for the int extent n of the square matrix A.
    n = A.value_shape[0]
    return total(
        reduce(Product, (Indexed(A, (r, p[r])) for r in range(n)), Integer(sign(p)))
        for p in permutations(range(n))
    )


def sign(p):
    return (-1) ** sum(p[a] > p[b] for a, b in combinations(range(len(p)), 2))
