from __future__ import annotations

from collections.abc import Callable, Iterable
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


def rank(f: AbstractExpression) -> int:
    return len(f.value_shape)


def add(first: AbstractExpression, second: AbstractExpression) -> AbstractExpression:
    from .tensors import component_tensor, indexed

    k = derived_indices("uflx.add", (first, second), rank(first))
    return component_tensor(Sum(indexed(first, k), indexed(second, k)), k)


def multiply(first: AbstractExpression, second: AbstractExpression) -> AbstractExpression:
    if rank(second) and not rank(first):
        k = derived_indices("uflx.multiply", (first, second), rank(second))
        return ComponentTensor(Product(first, Indexed(second, k)), k)
    if rank(first) and not rank(second):
        k = derived_indices("uflx.multiply", (first, second), rank(first))
        return ComponentTensor(Product(Indexed(first, k), second), k)
    return Product(first, second)


def divide(first: AbstractExpression, second: AbstractExpression) -> AbstractExpression:
    if rank(first):
        k = derived_indices("uflx.divide", (first, second), rank(first))
        return ComponentTensor(Division(Indexed(first, k), second), k)
    return Division(first, second)


def negate(f: AbstractExpression) -> AbstractExpression:
    return multiply(Integer(-1), f)


def subtract(f: AbstractExpression, g: AbstractExpression) -> AbstractExpression:
    return add(f, negate(g))


def power(first: AbstractExpression, second: AbstractExpression) -> AbstractExpression:
    return Power(first, second)


def absolute(f: AbstractExpression) -> AbstractExpression:
    return Absolute(f)


def conjugate(f: AbstractExpression) -> AbstractExpression:
    return Conjugate(f)


def entrywise(
    name: str, op: Callable[[AbstractExpression], AbstractExpression], f: AbstractExpression
) -> AbstractExpression:
    if not rank(f):
        return op(f)
    k = derived_indices(name, (f,), rank(f))
    return ComponentTensor(op(Indexed(f, k)), k)


def real_part(f: AbstractExpression) -> AbstractExpression:
    return entrywise("uflx.real_part", RealPart, f)


def imaginary_part(f: AbstractExpression) -> AbstractExpression:
    return entrywise("uflx.imaginary_part", ImaginaryPart, f)


def square_root(f: AbstractExpression) -> AbstractExpression:
    return SquareRoot(f)


def logarithm(f: AbstractExpression) -> AbstractExpression:
    return Logarithm(f)


def total(es: Iterable[AbstractExpression]) -> AbstractExpression:
    return reduce(Sum, es)
