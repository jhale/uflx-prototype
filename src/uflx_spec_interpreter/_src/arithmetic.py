from __future__ import annotations

from collections.abc import Callable, Iterable
from functools import reduce

from .names import _derived_indices
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

# Shared implementation helpers; the package public API is declared in __init__.py.
__all__ = [
    "_total",
    "absolute",
    "add",
    "conjugate",
    "divide",
    "imaginary_part",
    "logarithm",
    "multiply",
    "negate",
    "power",
    "real_part",
    "square_root",
    "subtract",
]


def add(first: AbstractExpression, second: AbstractExpression) -> AbstractExpression:
    from .tensors import component_tensor, indexed

    k = _derived_indices("uflx.add", (first, second), len(first.value_shape))
    return component_tensor(Sum(indexed(first, k), indexed(second, k)), k)


def multiply(first: AbstractExpression, second: AbstractExpression) -> AbstractExpression:
    if len(second.value_shape) and not len(first.value_shape):
        k = _derived_indices("uflx.multiply", (first, second), len(second.value_shape))
        return ComponentTensor(Product(first, Indexed(second, k)), k)
    if len(first.value_shape) and not len(second.value_shape):
        k = _derived_indices("uflx.multiply", (first, second), len(first.value_shape))
        return ComponentTensor(Product(Indexed(first, k), second), k)
    return Product(first, second)


def divide(first: AbstractExpression, second: AbstractExpression) -> AbstractExpression:
    if len(first.value_shape):
        k = _derived_indices("uflx.divide", (first, second), len(first.value_shape))
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


def _entrywise(
    name: str, op: Callable[[AbstractExpression], AbstractExpression], f: AbstractExpression
) -> AbstractExpression:
    if not len(f.value_shape):
        return op(f)
    k = _derived_indices(name, (f,), len(f.value_shape))
    return ComponentTensor(op(Indexed(f, k)), k)


def real_part(f: AbstractExpression) -> AbstractExpression:
    return _entrywise("uflx.real_part", RealPart, f)


def imaginary_part(f: AbstractExpression) -> AbstractExpression:
    return _entrywise("uflx.imaginary_part", ImaginaryPart, f)


def square_root(f: AbstractExpression) -> AbstractExpression:
    return SquareRoot(f)


def logarithm(f: AbstractExpression) -> AbstractExpression:
    return Logarithm(f)


def _total(es: Iterable[AbstractExpression]) -> AbstractExpression:
    return reduce(Sum, es)
