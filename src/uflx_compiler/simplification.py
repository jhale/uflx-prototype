import numpy as np

from uflx import determinant
from uflx.algorithms import complete_indices, map_fields, memoized, substitute_values
from uflx.nodes import (
    BroadcastTo,
    ComponentTensor,
    Conjugate,
    Division,
    EuclideanSpace,
    Float,
    GatheredIndex,
    ImaginaryPart,
    Index,
    Indexed,
    IndexSum,
    Integer,
    Integral,
    Inverse,
    Power,
    Product,
    RealPart,
    Sum,
    Tensor,
    Zero,
)


def simplify(f):
    # Rewrites that keep the value, so that each node keeps only the work and the free indices
    # that its value needs. A term that is zero is dropped, a component of a component tensor
    # is the expression at that component, and arithmetic of literals is folded.
    def walk(walk, e):
        new = map_fields(e, walk, e.value_fields)
        out = rewritten(new)
        # Finish the child before reconstructing a parent which binds its indices.
        return complete_indices(e, out if out is new else walk(out))

    return memoized(walk)(f)


def zero(e):
    return Zero(e.value_shape, e.free_indices, e.argument_spaces, e.scalar_type)


def literal(e):
    if isinstance(e, Indexed) and isinstance(e.expression, BroadcastTo):
        value = e.expression.value
        if not value.value_shape:
            return literal(value)
    if isinstance(e, Sum):
        if isinstance(e.first, Zero):
            return literal(e.second)
        if isinstance(e.second, Zero):
            return literal(e.first)
    return e.value if isinstance(e, Integer | Float) else None


def is_zero(e):
    return isinstance(e, Zero) or literal(e) == 0


def rewritten(e):
    match e:
        case Integral(g, EuclideanSpace(0)) if g.domain is None:
            # A point has weight one; tabulation has removed its spatial domain.
            return g
        case Sum(a, b) if is_zero(a) and b.scalar_type == e.scalar_type:
            return b
        case Sum(a, b) if is_zero(b) and a.scalar_type == e.scalar_type:
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


def instantiated(x, mapping):
    # Binding and free-index substitution have one implementation in the core.
    def fold(walk, e):
        e = map_fields(e, walk, (*e.value_fields, *e.index_fields))
        if isinstance(e, GatheredIndex) and all(type(i) is int for i in e.indices):
            from uflx_basix.nodes import IndexTable

            if isinstance(e.table, IndexTable):
                return int(np.asarray(e.table.values)[e.indices])
        return e

    return memoized(fold)(substitute_values(x, mapping))
