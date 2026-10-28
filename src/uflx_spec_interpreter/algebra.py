from functools import reduce
from itertools import combinations, permutations

from .domains import ambient_dim
from .nodes import (
    ComponentTensor,
    Coordinate,
    Division,
    Gradient,
    Index,
    Indexed,
    IndexSum,
    Num,
    Product,
    SquareRoot,
    Sum,
    Tensor,
)


def total(es):
    return reduce(Sum, es)


def subtract(b, c):
    return Sum(b, Product(Num(-1), c))


def identity(n):
    return Tensor(tuple(Tensor(tuple(Num(1 if r == c else 0) for c in range(n))) for r in range(n)))


def add(*Xs):
    r, s = (Index(("add", Xs, n)) for n in range(2))
    return ComponentTensor(total(Indexed(X, (r, s)) for X in Xs), (r, s))


def negate(X):
    r, s = (Index(("negate", X, n)) for n in range(2))
    return ComponentTensor(Product(Num(-1), Indexed(X, (r, s))), (r, s))


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
    return scale(Division(Num(1), norm(u)), u)


def gram(J):
    return matrix_product(transpose(J), J)


def tangent_gram(J, P):
    # JᵀJ + I - Π is invertible on a parameter domain of lower dimension, where JᵀJ is not.
    return add(gram(J), identity(ambient_dim(P)), negate(Gradient(Coordinate(P), P)))


def determinant(M, n):
    # The Leibniz formula.
    return total(
        reduce(Product, (Indexed(M, (r, p[r])) for r in range(n)), Num(sign(p)))
        for p in permutations(range(n))
    )


def sign(p):
    return (-1) ** sum(p[a] > p[b] for a, b in combinations(range(len(p)), 2))
