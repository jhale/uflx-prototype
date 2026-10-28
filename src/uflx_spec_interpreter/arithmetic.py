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
    return componentwise(lambda x: Product(Integer(-1), x), "negate", a)


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


def total(es):
    return reduce(Sum, es)
