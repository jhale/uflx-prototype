# Copyright (C) 2026 Jack S. Hale
#
# This file is part of uflx-core-experimental (https://fenics.github.io/uflx/language/)
#
# SPDX-License-Identifier:    MIT

from __future__ import annotations

from collections.abc import Callable, Iterable
from functools import reduce

from .algorithms.names import derived_indices
from .nodes import (
    Absolute,
    AbstractExpression,
    ComponentTensor,
    Conditional,
    Conjugate,
    Division,
    ImaginaryPart,
    Indexed,
    Integer,
    LessThan,
    Logarithm,
    Power,
    Product,
    RealPart,
    SquareRoot,
    Sum,
)


def add(first: AbstractExpression, second: AbstractExpression) -> AbstractExpression:
    return Sum(first, second)


def multiply(first: AbstractExpression, second: AbstractExpression) -> AbstractExpression:
    return Product(first, second)


def divide(numerator: AbstractExpression, denominator: AbstractExpression) -> AbstractExpression:
    return Division(numerator, denominator)


def negate(operand: AbstractExpression) -> AbstractExpression:
    return multiply(Integer(-1), operand)


def subtract(first: AbstractExpression, second: AbstractExpression) -> AbstractExpression:
    return add(first, negate(second))


def power(base: AbstractExpression, exponent: AbstractExpression) -> AbstractExpression:
    return Power(base, exponent)


def absolute(operand: AbstractExpression) -> AbstractExpression:
    return Absolute(operand)


def conjugate(operand: AbstractExpression) -> AbstractExpression:
    return Conjugate(operand)


def _entrywise(
    name: str, op: Callable[[AbstractExpression], AbstractExpression], f: AbstractExpression
) -> AbstractExpression:
    if f.value_shape == ():
        return op(f)
    k = derived_indices(name, (f,), len(f.value_shape))
    return ComponentTensor(op(Indexed(f, k)), k)


def real_part(operand: AbstractExpression) -> AbstractExpression:
    return _entrywise("uflx.real_part", RealPart, operand)


def imaginary_part(operand: AbstractExpression) -> AbstractExpression:
    return _entrywise("uflx.imaginary_part", ImaginaryPart, operand)


def square_root(operand: AbstractExpression) -> AbstractExpression:
    return SquareRoot(operand)


def logarithm(operand: AbstractExpression) -> AbstractExpression:
    return Logarithm(operand)


def total(es: Iterable[AbstractExpression]) -> AbstractExpression:
    return reduce(Sum, es)


def less_than(first: AbstractExpression, second: AbstractExpression) -> LessThan:
    return LessThan(first, second)


def conditional(
    predicate: AbstractExpression, true_value: AbstractExpression, false_value: AbstractExpression
) -> Conditional:
    return Conditional(predicate, true_value, false_value)
