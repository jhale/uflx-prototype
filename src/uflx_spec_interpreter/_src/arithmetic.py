from functools import reduce

from .nodes import (
    Absolute,
    ComponentTensor,
    Conjugate,
    Division,
    ImaginaryPart,
    Index,
    Indexed,
    Integer,
    Logarithm,
    Power,
    Product,
    RealPart,
    SquareRoot,
    Sum,
)


def derived(name, operands, n):
    return tuple(Index((name, *operands, m)) for m in range(n))


def rank(f):
    return len(f.value_shape)


def add(first, second):
    if rank(first) and rank(second):
        K = derived("uflx.add", (first, second), rank(first))
        return ComponentTensor(Sum(Indexed(first, K), Indexed(second, K)), K)
    return Sum(first, second)


def multiply(first, second):
    if rank(second) and not rank(first):
        K = derived("uflx.multiply", (first, second), rank(second))
        return ComponentTensor(Product(first, Indexed(second, K)), K)
    if rank(first) and not rank(second):
        K = derived("uflx.multiply", (first, second), rank(first))
        return ComponentTensor(Product(Indexed(first, K), second), K)
    return Product(first, second)


def divide(first, second):
    if rank(first) and not rank(second):
        K = derived("uflx.divide", (first, second), rank(first))
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
    K = derived(name, (f,), rank(f))
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
