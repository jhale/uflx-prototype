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


def subtract(first, second):
    return add(first, negate(second))


def power(a, b):
    return Power(a, b)


def absolute(a):
    return Absolute(a)


def conjugate(a):
    return Conjugate(a)


def entrywise(name, op, f):
    if not rank(f):
        return op(f)
    K = derived(name, (f,), rank(f))
    return ComponentTensor(op(Indexed(f, K)), K)


def real_part(f):
    return entrywise("uflx.real_part", RealPart, f)


def imaginary_part(f):
    return entrywise("uflx.imaginary_part", ImaginaryPart, f)


def square_root(a):
    return SquareRoot(a)


def logarithm(a):
    return Logarithm(a)


def total(es):
    return reduce(Sum, es)
