"""Numerical differentiation rules used by the interpreter's UFLx node cases."""

from dataclasses import dataclass
from functools import reduce
from itertools import count
from operator import add as _add
from operator import mul
from operator import sub as _sub

import numpy as np

_tags = count(1)


@dataclass(frozen=True, eq=False)
class Dual:
    """A value and one directional derivative. Tags distinguish nested derivatives."""

    tag: int
    value: object
    tangent: object


def fresh_tag():
    return next(_tags)


def level(value):
    return value.tag if isinstance(value, Dual) else 0


def primal(value):
    while isinstance(value, Dual):
        value = value.value
    return value


def pair(tag, value, tangent):
    newer = max(level(value), level(tangent))
    if newer > tag:
        v, dv = split(value, newer)
        t, dt = split(tangent, newer)
        return Dual(newer, pair(tag, v, t), pair(tag, dv, dt))
    return Dual(tag, value, tangent)


def split(value, tag):
    if isinstance(value, Dual):
        if value.tag == tag:
            return value.value, value.tangent
        if value.tag > tag:
            v, dv = split(value.value, tag)
            t, dt = split(value.tangent, tag)
            return Dual(value.tag, v, t), Dual(value.tag, dv, dt)
    return value, np.zeros_like(primal(value))


def tags(value):
    if isinstance(value, Dual):
        return {value.tag} | tags(value.value) | tags(value.tangent)
    return set()


def is_zero(value):
    if isinstance(value, Dual):
        return is_zero(value.value) and is_zero(value.tangent)
    return not callable(value) and bool(np.all(np.asarray(value) == 0))


def linear(fn, value):
    if isinstance(value, Dual):
        return Dual(value.tag, linear(fn, value.value), linear(fn, value.tangent))
    return fn(value)


def asarray(value, dtype=None):
    return linear(lambda x: np.asarray(x, dtype=dtype), value)


def entry(value, key):
    return linear(lambda x: x[key], value)


def stack(values, axis=0):
    tag = max(map(level, values), default=0)
    if tag:
        parts = [split(x, tag) for x in values]
        return Dual(tag, stack([v for v, _ in parts], axis), stack([t for _, t in parts], axis))
    return np.stack(values, axis=axis)


def combine(op, a, b):
    tag = max(level(a), level(b))
    if not tag:
        return op(a, b)
    av, da = split(a, tag)
    bv, db = split(b, tag)
    return Dual(tag, combine(op, av, bv), combine(op, da, db))


def add(a, b):
    return combine(_add, a, b)


def subtract(a, b):
    return combine(_sub, a, b)


def bilinear(fn, a, b):
    tag = max(level(a), level(b))
    if not tag:
        return fn(a, b)
    av, da = split(a, tag)
    bv, db = split(b, tag)
    tangent = bilinear(fn, da, bv) if level(a) == tag else 0
    if level(b) == tag:
        tangent = add(tangent, bilinear(fn, av, db))
    return Dual(tag, bilinear(fn, av, bv), tangent)


def multiply(a, b):
    return bilinear(mul, a, b)


def matmul(*matrices):
    return reduce(lambda a, b: bilinear(np.matmul, a, b), matrices)


def divide(a, b):
    tag = max(level(a), level(b))
    if not tag:
        return a / b
    av, da = split(a, tag)
    bv, db = split(b, tag)
    result = divide(av, bv)
    numerator = subtract(da, multiply(result, db)) if level(b) == tag else da
    return Dual(tag, result, divide(numerator, bv))


def power(a, b):
    tag = max(level(a), level(b))
    if not tag:
        return np.power(a, b)
    av, da = split(a, tag)
    bv, db = split(b, tag)
    result = power(av, bv)
    tangent = np.zeros_like(primal(result))
    if level(a) == tag and not is_zero(bv):
        tangent = multiply(multiply(bv, power(av, subtract(bv, 1))), da)
    if level(b) == tag:
        tangent = add(tangent, multiply(multiply(result, logarithm(av)), db))
    return Dual(tag, result, tangent)


def sqrt(a):
    if isinstance(a, Dual):
        root = sqrt(a.value)
        return Dual(a.tag, root, divide(a.tangent, multiply(2, root)))
    return np.sqrt(a)


def logarithm(a):
    if isinstance(a, Dual):
        return Dual(a.tag, logarithm(a.value), divide(a.tangent, a.value))
    return np.log(a)


def absolute(a):
    if isinstance(a, Dual):
        magnitude = absolute(a.value)
        dot = multiply(linear(np.conj, a.value), a.tangent)
        return Dual(a.tag, magnitude, divide(linear(np.real, dot), magnitude))
    return np.abs(a)


def determinant(matrix):
    if not isinstance(matrix, Dual):
        return np.linalg.det(matrix)
    a, da = matrix.value, matrix.tangent
    # Multilinearity in the rows also covers singular matrices.
    rows = range(len(primal(a)))
    tangent = 0
    for i in rows:
        row_matrix = stack([entry(da if j == i else a, j) for j in rows])
        tangent = add(tangent, determinant(row_matrix))
    return Dual(matrix.tag, determinant(a), tangent)


def inverse(matrix):
    if not isinstance(matrix, Dual):
        return np.linalg.inv(matrix)
    inv = inverse(matrix.value)
    return Dual(matrix.tag, inv, linear(np.negative, matmul(inv, matrix.tangent, inv)))


def pinv(matrix):
    if not isinstance(matrix, Dual):
        return np.linalg.pinv(matrix)
    a, da = matrix.value, matrix.tangent
    b = pinv(a)
    # Constant-rank derivative of the Moore-Penrose inverse.
    m, n = np.shape(primal(a))
    adjoint = lambda x: x.conj().T
    dah, bh = linear(adjoint, da), linear(adjoint, b)
    left = matmul(b, bh, dah, subtract(np.eye(m), matmul(a, b)))
    right = matmul(subtract(np.eye(n), matmul(b, a)), dah, bh, b)
    db = subtract(add(left, right), matmul(b, da, b))
    return Dual(matrix.tag, b, db)


def jacobian(fn, point):
    columns = []
    for axis in np.eye(len(primal(point))):
        tag = fresh_tag()
        result = fn(pair(tag, point, axis))
        columns.append(split(result, tag)[1])
    if not columns:
        return np.empty((*np.shape(primal(fn(point))), 0))
    return stack(columns, axis=-1)


def norm(value):
    squared = multiply(linear(np.conj, value), value)
    return sqrt(linear(lambda x: np.real(np.sum(x)), squared))
