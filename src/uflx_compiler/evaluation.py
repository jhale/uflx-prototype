import numpy as np

from uflx_spec_interpreter._src.nodes import (
    Absolute,
    ComponentTensor,
    Conjugate,
    Division,
    Float,
    GatheredIndex,
    ImaginaryPart,
    ImaginaryUnit,
    Index,
    Indexed,
    IndexSum,
    Integer,
    Inverse,
    Logarithm,
    Power,
    Product,
    RealPart,
    SquareRoot,
    Sum,
    Tensor,
    Zero,
)
from uflx_toy.nodes import RuntimeArray


def evaluate(e, env, xp=np):
    # Each node is an array with an axis for each of its free indices, in the order of its
    # static type, followed by the axes of its value shape.
    arrays = {}

    def value(n):
        if n not in arrays:
            arrays[n] = node(n, value, env, xp)
        return arrays[n]

    return value(e)


def free(n):
    return [i for i, _ in n.free_indices]


def extents(n):
    return dict(n.free_indices)


def aligned(n, data, order, xp):
    # The free axes of n moved into the given order, with an axis of length 1 for an absent index.
    axes = free(n)
    present = [i for i in order if i in axes]
    values = range(len(axes), len(axes) + len(n.value_shape))
    data = xp.transpose(data, [axes.index(i) for i in present] + list(values))
    lengths = dict(zip(present, data.shape, strict=False))
    return xp.reshape(data, [lengths.get(i, 1) for i in order] + list(data.shape[len(present) :]))


def node(n, value, env, xp):
    order = free(n)
    match n:
        case Integer(v) | Float(v):
            return xp.asarray(v)
        case ImaginaryUnit():
            return xp.asarray(1j)
        case Zero(value_shape, free_indices, _):
            return xp.zeros([m for _, m in free_indices] + list(value_shape))
        case RuntimeArray(label, _):
            return xp.asarray(env[label])
        case Sum(a, b):
            return aligned(a, value(a), order, xp) + aligned(b, value(b), order, xp)
        case Product(a, b):
            return aligned(a, value(a), order, xp) * aligned(b, value(b), order, xp)
        case Division(a, b):
            return aligned(a, value(a), order, xp) / aligned(b, value(b), order, xp)
        case Power(a, b):
            return xp.power(aligned(a, value(a), order, xp), aligned(b, value(b), order, xp))
        case (
            Absolute(a)
            | SquareRoot(a)
            | Logarithm(a)
            | Conjugate(a)
            | RealPart(a)
            | ImaginaryPart(a)
        ):
            op = {
                Absolute: xp.abs,
                SquareRoot: xp.sqrt,
                Logarithm: xp.log,
                Conjugate: xp.conj,
                RealPart: xp.real,
                ImaginaryPart: xp.imag,
            }[type(n)]
            return op(value(a))
        case Inverse(a):
            return xp.linalg.inv(value(a))
        case Tensor(entries):
            parts = [
                xp.broadcast_to(
                    aligned(x, value(x), order, xp),
                    full_shape(n, order)[: len(order)] + list(x.value_shape),
                )
                for x in entries
            ]
            return xp.stack(parts, axis=len(order))
        case Indexed(x, indices):
            return indexed(n, x, indices, value, xp)
        case ComponentTensor(x, indices):
            return xp.reshape(aligned(x, value(x), order + list(indices), xp), full_shape(n, order))
        case IndexSum(x, indices):
            data = aligned(x, value(x), order + list(indices), xp)
            return xp.sum(data, axis=tuple(range(len(order), len(order) + len(indices))))
    raise TypeError(f"the compiler cannot evaluate {type(n).__name__}")


def full_shape(n, order):
    return [extents(n)[i] for i in order] + list(n.value_shape)


def indexed(n, x, indices, value, xp):
    # Every axis of x is read through an integer array that broadcasts over the free indices.
    order = free(n)
    sizes = extents(n)

    def iota(i):
        return np.arange(sizes[i]).reshape([-1 if j == i else 1 for j in order])

    def entry(k):
        match k:
            case Index():
                return iota(k)
            case GatheredIndex(table, entries):
                return np.asarray(table.values)[tuple(entry(m) for m in entries)]
        return k

    keys = [iota(i) for i in free(x)] + [entry(k) for k in indices]
    return value(x)[tuple(keys)]
