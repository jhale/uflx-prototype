import numpy as np

from uflx_spec_interpreter._src.nodes import (
    Absolute,
    AbstractExtent,
    ComponentTensor,
    Conjugate,
    Coordinate,
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


def evaluate(e, env, xp=np, folded=None):
    # Each node is an array with an axis for each of its free indices, in the order of its
    # static type, followed by the axes of its value shape. A folded node is a constant.
    arrays = {n: xp.asarray(v) for n, v in (folded or {}).items()}

    def value(n):
        if n not in arrays:
            arrays[n] = node(n, value, env, xp)
        return arrays[n]

    return value(e)


def free(n):
    return [i for i, _ in n.free_indices]


def size(m, env):
    # An extent known only at assembly is a kernel input, as a solver gives it.
    return env[m.label] if isinstance(m, AbstractExtent) else m


def extents(n, env):
    return {i: size(m, env) for i, m in n.free_indices}


def value_shape(n, env):
    return [size(m, env) for m in n.value_shape]


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
        case Zero(shape, free_indices, _):
            return xp.zeros([size(m, env) for _, m in free_indices] + [size(m, env) for m in shape])
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
        case Coordinate(P) if P.dimension == 0:
            return xp.zeros((0,))
        case Inverse(a):
            return xp.linalg.inv(value(a))
        case Tensor(entries):
            parts = [
                xp.broadcast_to(
                    aligned(x, value(x), order, xp),
                    full_shape(n, order, env)[: len(order)] + value_shape(x, env),
                )
                for x in entries
            ]
            return xp.stack(parts, axis=len(order))
        case Indexed(x, indices):
            return indexed(n, x, indices, value, env, xp)
        case ComponentTensor(x, indices):
            return xp.reshape(
                aligned(x, value(x), order + list(indices), xp), full_shape(n, order, env)
            )
        case IndexSum(x, indices):
            data = aligned(x, value(x), order + list(indices), xp)
            return xp.sum(data, axis=tuple(range(len(order), len(order) + len(indices))))
    raise TypeError(f"the compiler cannot evaluate {type(n).__name__}")


def full_shape(n, order, env):
    return [extents(n, env)[i] for i in order] + value_shape(n, env)


def indexed(n, x, indices, value, env, xp):
    # Every axis of x is read through an integer array that broadcasts over the free indices.
    order = free(n)
    sizes = extents(n, env)

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
