from functools import reduce
from itertools import combinations, permutations

from .domains import ambient_dim
from .nodes import (
    CT,
    Coordinate,
    Div,
    Gradient,
    Index,
    Indexed,
    Inv,
    ISum,
    Num,
    Prod,
    Sqrt,
    Sum,
    Tensor,
)


def total(es):
    return reduce(Sum, es)


def minus(b, c):
    return Sum(b, Prod(Num(-1), c))


def identity(n):
    return Tensor(tuple(Tensor(tuple(Num(1 if r == c else 0) for c in range(n))) for r in range(n)))


def add(*Xs):
    r, s = (Index(("add", Xs, n)) for n in range(2))
    return CT(total(Indexed(X, (r, s)) for X in Xs), (r, s))


def negate(X):
    r, s = (Index(("negate", X, n)) for n in range(2))
    return CT(Prod(Num(-1), Indexed(X, (r, s))), (r, s))


def transpose(X):
    r, s = (Index(("transpose", X, n)) for n in range(2))
    return CT(Indexed(X, (s, r)), (r, s))


def matmul(X, Y, *rest):
    r, s, t = (Index(("matmul", X, Y, n)) for n in range(3))
    XY = CT(ISum(Prod(Indexed(X, (r, t)), Indexed(Y, (t, s))), (t,)), (r, s))
    return matmul(XY, *rest) if rest else XY


def matvec(X, u):
    r, t = (Index(("matvec", X, u, n)) for n in range(2))
    return CT(ISum(Prod(Indexed(X, (r, t)), Indexed(u, (t,))), (t,)), (r,))


def dot(u, v):
    r = Index(("dot", u, v))
    return ISum(Prod(Indexed(u, (r,)), Indexed(v, (r,))), (r,))


def scale(c, u):
    r = Index(("scale", c, u))
    return CT(Prod(c, Indexed(u, (r,))), (r,))


def norm(u):
    return Sqrt(dot(u, u))


def unit(u):
    return scale(Div(Num(1), norm(u)), u)


def gram(J):
    return matmul(transpose(J), J)


def tangent_gram(J, P):
    # JᵀJ + I - Π is invertible on a parameter domain of lower dimension, where JᵀJ is not.
    return add(gram(J), identity(ambient_dim(P)), negate(Gradient(Coordinate(P), P)))


def tangent_inverse(J, P):
    # (JᵀJ + I - Π)⁻¹ Jᵀ is the pseudo-inverse of J on the tangent space of P.
    return matmul(Inv(tangent_gram(J, P)), transpose(J))


def determinant(M, n):
    # The Leibniz formula.
    return total(
        reduce(Prod, (Indexed(M, (r, p[r])) for r in range(n)), Num(sign(p)))
        for p in permutations(range(n))
    )


def sign(p):
    return (-1) ** sum(p[a] > p[b] for a, b in combinations(range(len(p)), 2))
