import numpy as np

from uflx_spec_interpreter._src.names import derived_index
from uflx_spec_interpreter._src.nodes import (
    ComponentTensor,
    Conjugate,
    Division,
    EllipsisIndex,
    Float,
    GatheredIndex,
    ImaginaryPart,
    Index,
    Indexed,
    IndexSum,
    Integer,
    Inverse,
    Power,
    Product,
    RealPart,
    Sum,
    Tensor,
    Zero,
)
from uflx_spec_interpreter._src.substitution import lowered, map_fields, memoized
from uflx_spec_interpreter._src.tensors import determinant


def simplify(f):
    # Rewrites that keep the value, so that each node keeps only the work and the free indices
    # that its value needs. A term that is zero is dropped, a component of a component tensor
    # is the expression at that component, and arithmetic of literals is folded.
    def walk(walk, e):
        new = map_fields(e, walk, lowered(e))
        if isinstance(e, ComponentTensor | IndexSum):
            new = restored(e, new)
        out = rewritten(new)
        return out if out is new else walk(out)

    return memoized(walk)(f)


def zero(e):
    return Zero(e.value_shape, e.free_indices, e.argument_spaces)


def literal(e):
    return e.value if isinstance(e, Integer | Float) else None


def is_zero(e):
    return isinstance(e, Zero) or literal(e) == 0


def rewritten(e):
    match e:
        case Sum(a, b) if is_zero(a):
            return b
        case Sum(a, b) if is_zero(b):
            return a
        case Product(a, b) if is_zero(a) or is_zero(b):
            return zero(e)
        case Division(a, _) if is_zero(a):
            return zero(e)
        case Indexed(Zero(), _) | ComponentTensor(Zero(), _) | IndexSum(Zero(), _):
            return zero(e)
        case Conjugate(Zero()) | RealPart(Zero()) | ImaginaryPart(Zero()):
            return zero(e)
        case Tensor(entries) if all(isinstance(x, Zero) for x in entries):
            return zero(e)
        case Product(a, b) if literal(a) == 1:
            return b
        case Product(a, b) if literal(b) == 1:
            return a
        case Division(a, b) | Power(a, b) if literal(b) == 1:
            return a
        case Sum(a, b) if None not in (literal(a), literal(b)):
            return folded(literal(a) + literal(b), a, b)
        case Product(a, b) if None not in (literal(a), literal(b)):
            return folded(literal(a) * literal(b), a, b)
        case Division(a, b) if None not in (literal(a), literal(b)):
            return folded(literal(a) / literal(b), a, b)
        case Indexed(ComponentTensor(x, bound), indices) if reducible(bound, indices):
            return instantiated(x, dict(zip(bound, indices, strict=True)))
        case Indexed(Tensor(entries), (int() as k, *rest)):
            return Indexed(entries[k], tuple(rest)) if rest else entries[k]
        case Inverse(a) if a.value_shape in ((2, 2), (3, 3)):
            return adjugate(a)
    return e


def folded(value, a, b):
    if isinstance(a, Integer) and isinstance(b, Integer) and type(value) is int:
        return Integer(value)
    return Float(float(value))


def reducible(bound, indices):
    simple = (Index, int)
    return all(isinstance(i, Index) for i in bound) and all(isinstance(k, simple) for k in indices)


def adjugate(A):
    # The inverse as the adjugate over the determinant, which is the node of the density too.
    n, det = A.value_shape[0], determinant(A)

    def cofactor(r, c):
        rows = [m for m in range(n) if m != r]
        cols = [m for m in range(n) if m != c]
        minor = Tensor(tuple(Tensor(tuple(Indexed(A, (i, j)) for j in cols)) for i in rows))
        return Product(Integer((-1) ** (r + c)), determinant(minor))

    return Tensor(
        tuple(Tensor(tuple(Division(cofactor(c, r), det) for c in range(n))) for r in range(n))
    )


def restored(e, new):
    # A bound index that the operand lost comes back through a Zero, which broadcasts there.
    extents, kept = dict(e.expression.free_indices), dict(new.expression.free_indices)
    lost = tuple((i, extents[i]) for i in e.indices if i not in kept)
    if not lost:
        return new
    x = new.expression
    return type(e)(Sum(x, Zero(x.value_shape, lost, x.argument_spaces)), e.indices)


def instantiated(x, mapping):
    # x with each Index of mapping replaced by an Index or an int, through the free indices of
    # a Zero too, and with the indices of a binding node renamed where they would capture.
    def walk(walk, e, items):
        mapping = dict(items)
        match e:
            case Index():
                return mapping.get(e, e)
            case GatheredIndex(table, indices):
                indices = tuple(
                    walk(i, items) if isinstance(i, Index | GatheredIndex) else i for i in indices
                )
                if all(isinstance(i, int) for i in indices):
                    return int(np.asarray(table.values)[indices])
                return GatheredIndex(table, indices)
            case Zero(shape, free, spaces):
                return Zero(shape, tuple(p for i, n in free for p in moved(i, n, mapping)), spaces)
            case ComponentTensor(body, bound) | IndexSum(body, bound):
                inner = {i: k for i, k in mapping.items() if i not in bound}
                captured = [i for i in bound if i in inner.values()]
                renames = {i: derived_index("compiler.simplify", e, i) for i in captured}
                if renames:
                    body = walk(body, tuple(renames.items()))
                    bound = tuple(renames.get(i, i) for i in bound)
                return type(e)(walk(body, tuple(inner.items())), bound)
        return map_fields(e, lambda c: walk(c, items))

    return memoized(walk)(x, tuple(mapping.items()))


def moved(i, n, mapping):
    k = mapping.get(i, i)
    if isinstance(k, Index | EllipsisIndex):
        return ((k, n),)
    if isinstance(k, GatheredIndex):
        return k.free_indices
    return ()
