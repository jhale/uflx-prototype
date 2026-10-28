from functools import reduce
from itertools import combinations, permutations

from .nodes import (
    Absolute,
    ComponentTensor,
    Conjugate,
    Coordinate,
    Division,
    GatheredIndex,
    Gradient,
    ImaginaryPart,
    Index,
    Indexed,
    IndexSum,
    Inverse,
    Logarithm,
    Num,
    Power,
    Product,
    RealPart,
    SquareRoot,
    Sum,
    Tensor,
    Zero,
)


def num(value):
    return Num(value)


def zero(value_shape, free_indices, argument_spaces):
    return Zero(value_shape, free_indices, argument_spaces)


def index(label):
    return Index(label)


def gathered_index(table, indices):
    return GatheredIndex(table, indices)


def tensor(entries):
    return Tensor(tuple(entries))


def indexed(expression, indices):
    return Indexed(expression, tuple(indices))


def component_tensor(expression, indices):
    return ComponentTensor(expression, tuple(indices))


def index_sum(expression, indices):
    return IndexSum(expression, tuple(indices))


def componentwise(rule, label, *operands):
    # The rule on the components, through derived indices, for an array-valued operand.
    shape = operands[0].value_shape
    if not shape:
        return rule(*operands)
    axes = tuple(Index((label, operands, n)) for n in range(len(shape)))
    return ComponentTensor(rule(*(Indexed(x, axes) for x in operands)), axes)


def add(a, b):
    return componentwise(Sum, "add", a, b)


def negate(a):
    return componentwise(lambda x: Product(Num(-1), x), "negate", a)


def subtract(a, b):
    return add(a, negate(b))


def multiply(a, b):
    return Product(a, b)


def divide(a, b):
    return Division(a, b)


def power(a, b):
    return Power(a, b)


def absolute(a):
    return Absolute(a)


def conjugate(a):
    return Conjugate(a)


def real_part(a):
    return RealPart(a)


def imaginary_part(a):
    return ImaginaryPart(a)


def square_root(a):
    return SquareRoot(a)


def logarithm(a):
    return Logarithm(a)


def inverse(a):
    return Inverse(a)


def total(es):
    return reduce(Sum, es)


def identity(n):
    return Tensor(tuple(Tensor(tuple(Num(1 if r == c else 0) for c in range(n))) for r in range(n)))


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
    identity_matrix = identity(P.ambient_coordinate_domain.dimension)
    return subtract(add(gram(J), identity_matrix), Gradient(Coordinate(P), P))


def determinant(M, n):
    # The Leibniz formula.
    return total(
        reduce(Product, (Indexed(M, (r, p[r])) for r in range(n)), Num(sign(p)))
        for p in permutations(range(n))
    )


def sign(p):
    return (-1) ** sum(p[a] > p[b] for a, b in combinations(range(len(p)), 2))
