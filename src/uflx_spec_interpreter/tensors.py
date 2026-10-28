from functools import reduce
from itertools import combinations, permutations

from .arithmetic import total
from .nodes import (
    ComponentTensor,
    Division,
    Index,
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


def inverse(a):
    return Inverse(a)


def identity(n):
    return Tensor(
        tuple(Tensor(tuple(Integer(1 if r == c else 0) for c in range(n))) for r in range(n))
    )


def transpose(X):
    r, s = (Index(("transpose", X, n)) for n in range(2))
    return ComponentTensor(Indexed(X, (s, r)), (r, s))


def matrix_product(X, Y, *rest):
    r, s, t = (Index(("matmul", X, Y, n)) for n in range(3))
    XY = ComponentTensor(IndexSum(Product(Indexed(X, (r, t)), Indexed(Y, (t, s))), (t,)), (r, s))
    return matrix_product(XY, *rest) if rest else XY


def matvec(X, u):
    r, t = (Index(("matvec", X, u, n)) for n in range(2))
    return ComponentTensor(IndexSum(Product(Indexed(X, (r, t)), Indexed(u, (t,))), (t,)), (r,))


def dot(u, v):
    r = Index(("dot", u, v))
    return IndexSum(Product(Indexed(u, (r,)), Indexed(v, (r,))), (r,))


def scale(c, u):
    r = Index(("scale", c, u))
    return ComponentTensor(Product(c, Indexed(u, (r,))), (r,))


def norm(u):
    return SquareRoot(dot(u, u))


def unit(u):
    return scale(Division(Integer(1), norm(u)), u)


def determinant(M, n):
    # The Leibniz formula.
    return total(
        reduce(Product, (Indexed(M, (r, p[r])) for r in range(n)), Integer(sign(p)))
        for p in permutations(range(n))
    )


def sign(p):
    return (-1) ** sum(p[a] > p[b] for a, b in combinations(range(len(p)), 2))
