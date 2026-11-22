"""Tagged forward-mode differentiation of numerical values, including arrays.

Each derivative introduces a fresh tag. Keeping the tags distinct allows a field
captured by an outer derivative to be evaluated inside another derivative.
"""

from dataclasses import dataclass
from itertools import count

import numpy as np

_tags = count(1)


def fresh_tag():
    return next(_tags)


def level(value):
    return value.tag if isinstance(value, Dual) else 0


def primal(value):
    while isinstance(value, Dual):
        value = value.value
    return value


def pair(tag, value, tangent):
    """Seed a direction, keeping newer differentiation levels on the outside."""
    value, tangent = normalize(value), normalize(tangent)
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
    if isinstance(value, tuple):
        return set().union(*(tags(v) for v in value))
    if callable(value) and hasattr(value, "_derivative_tags"):
        return value._derivative_tags()
    return set()


def is_zero(value):
    if isinstance(value, Dual):
        return is_zero(value.value) and is_zero(value.tangent)
    return not callable(value) and bool(np.all(np.asarray(value) == 0))


def normalize(value):
    if isinstance(value, (list, tuple)) or (
        isinstance(value, np.ndarray) and value.dtype == object
    ):
        return asarray(value)
    return value


def call_field(field, point):
    if isinstance(point, Dual) and hasattr(field, "jvp"):
        return pair(
            point.tag, call_field(field, point.value), field.jvp(point.value, point.tangent)
        )
    return normalize(field(point))


def linear(fn, value):
    if isinstance(value, Dual):
        return Dual(value.tag, linear(fn, value.value), linear(fn, value.tangent))
    return fn(value)


def asarray(value, dtype=None):
    """Collect arrays of traced scalar entries returned by Python field callbacks."""
    if isinstance(value, Dual):
        return linear(lambda x: np.asarray(x, dtype=dtype), value) if dtype else value
    array = np.asarray(value)
    if array.dtype == object and any(isinstance(x, Dual) for x in array.flat):
        entries = stack(list(array.flat))
        collected = entries.reshape((*array.shape, *entries.shape[1:]))
        return asarray(collected, dtype) if dtype is not None else collected
    return np.asarray(value, dtype=dtype) if dtype else array


def stack(values, axis=0):
    tag = max(map(level, values), default=0)
    if tag:
        parts = [split(x, tag) for x in values]
        return Dual(tag, stack([v for v, _ in parts], axis), stack([t for _, t in parts], axis))
    return np.stack(values, axis=axis)


def bilinear(fn, a, b):
    tag = max(level(a), level(b))
    if not tag:
        return fn(a, b)
    av, da = split(a, tag)
    bv, db = split(b, tag)
    tangent = bilinear(fn, da, bv) if level(a) == tag else 0
    if level(b) == tag:
        tangent = tangent + bilinear(fn, av, db)
    return Dual(tag, bilinear(fn, av, bv), tangent)


def determinant(matrix):
    if not isinstance(matrix, Dual):
        return np.linalg.det(matrix)
    a, da = matrix.value, matrix.tangent
    # Multilinearity in the rows also covers singular matrices.
    tangent = sum(
        determinant(stack([da[j] if j == i else a[j] for j in range(matrix.shape[0])]))
        for i in range(matrix.shape[0])
    )
    return Dual(matrix.tag, determinant(a), tangent)


def inverse(matrix):
    if not isinstance(matrix, Dual):
        return np.linalg.inv(matrix)
    inv = inverse(matrix.value)
    return Dual(matrix.tag, inv, -inv @ matrix.tangent @ inv)


def pinv(matrix):
    if not isinstance(matrix, Dual):
        return np.linalg.pinv(matrix)
    a, da = matrix.value, matrix.tangent
    b = pinv(a)
    # Constant-rank derivative of the Moore-Penrose inverse.
    dah, bh = da.conj().T, b.conj().T
    db = -b @ da @ b + b @ bh @ dah @ (np.eye(a.shape[0]) - a @ b)
    db = db + (np.eye(a.shape[1]) - b @ a) @ dah @ bh @ b
    return Dual(matrix.tag, b, db)


def jacobian(fn, point):
    cols = []
    for axis in np.eye(np.shape(primal(point))[0]):
        tag = fresh_tag()
        result = asarray(fn(pair(tag, point, axis)))
        cols.append(split(result, tag)[1])
    if not cols:
        return np.empty((*np.shape(primal(fn(point))), 0))
    return stack(cols, axis=-1)


def norm(value):
    return np.sqrt(np.real(np.sum(np.conj(value) * value)))


@dataclass(frozen=True, eq=False)
class Dual(np.lib.mixins.NDArrayOperatorsMixin):
    tag: int
    value: object
    tangent: object

    @property
    def shape(self):
        return np.shape(primal(self))

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def size(self):
        return np.size(primal(self))

    def __getitem__(self, key):
        return linear(lambda x: np.asarray(x)[key], self)

    def reshape(self, *shape):
        return linear(lambda x: np.reshape(x, shape[0] if len(shape) == 1 else shape), self)

    @property
    def T(self):
        return linear(np.transpose, self)

    @property
    def real(self):
        return linear(np.real, self)

    @property
    def imag(self):
        return linear(np.imag, self)

    def conj(self):
        return linear(np.conj, self)

    def __bool__(self):
        return bool(primal(self))

    def __float__(self):
        raise TypeError(
            "field callback discarded a derivative; supply a jvp or an external derivative action"
        )

    __complex__ = __float__

    def __array_ufunc__(self, op, method, *inputs, **kwargs):
        if method != "__call__" or kwargs:
            return NotImplemented
        if op in (np.multiply, np.matmul):
            return bilinear(op, *inputs)
        if len(inputs) == 1:
            a, da = self.value, self.tangent
            if op in (np.negative, np.positive, np.conjugate):
                return linear(op, self)
            result = op(a)
            if op is np.sqrt:
                tangent = da / (2 * result)
            elif op is np.log:
                tangent = da / a
            elif op is np.absolute:
                tangent = np.real(np.conj(a) * da) / result
            elif op is np.sin:
                tangent = np.cos(a) * da
            elif op is np.cos:
                tangent = -np.sin(a) * da
            elif op is np.exp:
                tangent = result * da
            elif op is np.square:
                tangent = 2 * a * da
            else:
                return NotImplemented
            return Dual(self.tag, result, tangent)

        a, b = inputs
        tag = max(level(a), level(b))
        av, da = split(a, tag)
        bv, db = split(b, tag)
        if op in (np.less, np.less_equal, np.greater, np.greater_equal, np.equal, np.not_equal):
            return op(primal(av), primal(bv))
        result = op(av, bv)
        if op in (np.add, np.subtract):
            tangent = op(da, db)
        elif op is np.divide:
            tangent = (da - result * db) / bv if level(b) == tag else da / bv
        elif op in (np.power, np.float_power):
            if level(a) != tag:
                tangent = 0
            elif not level(bv) and np.all(bv == 0):
                tangent = np.zeros_like(primal(result))
            else:
                tangent = bv * (av ** (bv - 1)) * da
            if level(b) == tag:
                tangent = tangent + result * np.log(av) * db
        else:
            return NotImplemented
        return Dual(tag, result, tangent)

    def __array_function__(self, func, types, args, kwargs):
        if func in (
            np.real,
            np.imag,
            np.sum,
            np.trace,
            np.moveaxis,
            np.reshape,
            np.transpose,
            np.broadcast_to,
        ):
            return linear(lambda x: func(x, *args[1:], **kwargs), args[0])
        if func is np.stack:
            return stack(*args, **kwargs)
        if func in (np.dot, np.tensordot):
            return bilinear(lambda a, b: func(a, b, *args[2:], **kwargs), *args[:2])
        if func in (np.linalg.det, np.linalg.inv, np.linalg.pinv):
            return {np.linalg.det: determinant, np.linalg.inv: inverse, np.linalg.pinv: pinv}[func](
                args[0]
            )
        if func is np.linalg.norm and len(args) == 1 and not kwargs:
            return norm(args[0])
        return NotImplemented
