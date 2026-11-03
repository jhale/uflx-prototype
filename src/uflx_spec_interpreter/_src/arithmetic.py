from functools import reduce

from .names import derived_indices
from .nodes import (
    Absolute,
    AbstractExpression,
    ComponentTensor,
    Conjugate,
    Division,
    ImaginaryPart,
    Indexed,
    Integer,
    Logarithm,
    Power,
    Product,
    RealPart,
    SquareRoot,
    Sum,
)


def rank(f):
    if not isinstance(f, AbstractExpression):
        raise TypeError("operand must be an expression")
    return len(f.value_shape)


def add(first, second):
    from .tensors import component_tensor, indexed

    K = derived_indices("uflx.add", (first, second), rank(first))
    return component_tensor(Sum(indexed(first, K), indexed(second, K)), K)


def multiply(first, second):
    if rank(second) and not rank(first):
        K = derived_indices("uflx.multiply", (first, second), rank(second))
        return ComponentTensor(Product(first, Indexed(second, K)), K)
    if rank(first) and not rank(second):
        K = derived_indices("uflx.multiply", (first, second), rank(first))
        return ComponentTensor(Product(Indexed(first, K), second), K)
    return Product(first, second)


def divide(first, second):
    if rank(first):
        K = derived_indices("uflx.divide", (first, second), rank(first))
        return ComponentTensor(Division(Indexed(first, K), second), K)
    return Division(first, second)


def negate(f):
    return multiply(Integer(-1), f)


def subtract(f, g):
    return add(f, negate(g))


def power(first, second):
    return Power(first, second)


def absolute(f):
    return Absolute(f)


def conjugate(f):
    return Conjugate(f)


def entrywise(name, op, f):
    if not rank(f):
        return op(f)
    K = derived_indices(name, (f,), rank(f))
    return ComponentTensor(op(Indexed(f, K)), K)


def real_part(f):
    return entrywise("uflx.real_part", RealPart, f)


def imaginary_part(f):
    return entrywise("uflx.imaginary_part", ImaginaryPart, f)


def square_root(f):
    return SquareRoot(f)


def logarithm(f):
    return Logarithm(f)


def total(es):
    return reduce(Sum, es)
