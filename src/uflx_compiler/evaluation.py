import string

import numpy as np

from uflx._src.nodes import (
    Absolute,
    AbstractExtent,
    BroadcastTo,
    Complex,
    ComponentTensor,
    Conjugate,
    Coordinate,
    Determinant,
    Division,
    Float,
    GatheredIndex,
    ImaginaryPart,
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
from uflx_basix.nodes import RuntimeArray


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
    # The free axes of n moved into the given order, with an axis of length 1 for an absent
    # index. With n free in (i, j) and the order (j, k, i), the axes become (j, 1, i).
    axes = free(n)
    present = [i for i in order if i in axes]
    values = range(len(axes), len(axes) + len(n.value_shape))
    data = xp.transpose(data, [axes.index(i) for i in present] + list(values))
    lengths = dict(zip(present, data.shape, strict=False))
    return xp.reshape(data, [lengths.get(i, 1) for i in order] + list(data.shape[len(present) :]))


def node(n, value, env, xp):
    order = free(n)

    def operand(x, axes=order):
        # The value of x with its free axes in the given order, to broadcast against.
        return aligned(x, value(x), axes, xp)

    match n:
        case Integer(v) | Float(v):
            return xp.asarray(v)
        case Complex(a, b):
            return xp.asarray(complex(a.value, b.value))
        case Zero():
            return xp.zeros(
                full_shape(n, order, env),
                dtype=complex if n.scalar_type == "complex" else float,
            )
        case RuntimeArray(label, _):
            return xp.asarray(env[label])
        case Sum(a, b):
            return operand(a) + operand(b)
        case Product(a, b):
            return operand(a) * operand(b)
        case Division(a, b):
            return operand(a) / operand(b)
        case Power(a, b):
            return operand(a) ** operand(b)
        case Absolute(a):
            return xp.abs(value(a))
        case SquareRoot(a):
            return xp.sqrt(value(a))
        case Logarithm(a):
            return xp.log(value(a))
        case Conjugate(a):
            return xp.conj(value(a))
        case RealPart(a):
            return xp.real(value(a))
        case ImaginaryPart(a):
            return xp.imag(value(a))
        case Coordinate(P) if P.dimension == 0:
            return xp.zeros((0,))
        case Determinant(a):
            return xp.linalg.det(value(a))
        case Inverse(a) if a.value_shape[0] == a.value_shape[1]:
            return xp.linalg.inv(value(a))
        case Inverse(a):
            # A matrix with more rows than columns has the pseudo-inverse of the spec.
            A = value(a)
            At = xp.swapaxes(A, -1, -2)
            return xp.linalg.inv(At @ A) @ At
        case BroadcastTo(x, _):
            shape = full_shape(n, order, env)
            if 0 in shape:
                return xp.empty(shape, dtype=complex if n.scalar_type == "complex" else float)
            leading = len(n.value_shape) - len(x.value_shape)
            source = operand(x)
            expanded = source.shape[: len(order)] + (1,) * leading + source.shape[len(order) :]
            return xp.broadcast_to(xp.reshape(source, expanded), shape)
        case Tensor(entries):
            # The entries, each over every free index of n, stacked as the first axis of the value.
            lengths = [extents(n, env)[i] for i in order]
            parts = [xp.broadcast_to(operand(x), lengths + value_shape(x, env)) for x in entries]
            return xp.stack(parts, axis=len(order))
        case Indexed(x, indices):
            return indexed(n, x, indices, value, env, xp)
        case ComponentTensor(x, indices):
            return operand(x, order + list(indices))
        case IndexSum(x, indices) if any(size(dict(x.free_indices)[i], env) == 0 for i in indices):
            return xp.zeros(
                full_shape(n, order, env),
                dtype=complex if n.scalar_type == "complex" else float,
            )
        case IndexSum(Product() as x, indices):
            return contracted(n, x, value, xp)
        case IndexSum(x, indices):
            summed = tuple(range(len(order), len(order) + len(indices)))
            return xp.sum(operand(x, order + list(indices)), axis=summed)
    raise TypeError(f"the compiler cannot evaluate {type(n).__name__}")


def contracted(n, x, value, xp):
    # One einsum, whose path orders the sums, such as sum factorisation for factored tables.
    terms, letters = factors(x), {}

    def subscripts(m):
        return "".join(letters.setdefault(i, LETTERS[len(letters)]) for i in free(m))

    spec = ",".join(map(subscripts, terms)) + "->" + subscripts(n)
    return xp.einsum(spec, *map(value, terms), optimize=True)


def factors(x):
    return factors(x.first) + factors(x.second) if isinstance(x, Product) else [x]


LETTERS = string.ascii_letters


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
            case GatheredIndex() as g:
                return np.asarray(g.table.values)[tuple(entry(m) for m in g.indices)]
        return k

    keys = [iota(i) for i in free(x)] + [entry(k) for k in indices]
    return value(x)[tuple(keys)]
