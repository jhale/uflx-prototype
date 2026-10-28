from ..algebra import identity
from ..calculus import derivative_of_node
from ..domains import ambient_dim, topdim
from ..nodes import CT, Coordinate, Fn, Gradient, Index, Indexed, Normal, PullBack, Space, Zero
from ..static import arity, free_indices, shape, support
from ..substitution import expressions, map_fields
from .pullbacks import expand_pull_backs

# Expressions whose gradient a library provides, so the gradient remains.
LEAVES = (Fn, Normal, PullBack, Gradient, Coordinate)


def partial(f, P, j):
    # Entry j of the last axis of Gradient(f, P), with the leaf rows of apply_gradients first.
    match f:
        case _ if support(f) is None:
            return Zero(shape(f), (*free_indices(f), (j, ambient_dim(P))), zero_args(f))
        case Coordinate(Q) if Q == P and topdim(P) == ambient_dim(P):
            i = Index(("uflx.apply_gradients", f, P))
            return CT(Indexed(identity(ambient_dim(P)), (i, j)), (i,))
        case _ if isinstance(f, LEAVES):
            axes = tuple(Index(("uflx.apply_gradients", f, P, n)) for n in range(len(shape(f))))
            entry = Indexed(Gradient(f, P), (*axes, j))
            return CT(entry, axes) if axes else entry
    return derivative_of_node(f, lambda b: partial(b, P, j), lambda c: support(c) is None, [])


def zero_args(f):
    return tuple(Space("arg") for _ in range(arity(f)))


def gradient(f, P):
    j = Index(("uflx.apply_gradients", f, P, "j"))
    axes = tuple(Index(("uflx.apply_gradients", f, P, n)) for n in range(len(shape(f))))
    df = partial(f, P, j)
    return CT(Indexed(df, axes) if axes else df, (*axes, j))


def evaluated(f, P):
    # A gradient remains on a leaf, except the gradient of the coordinates, which is the identity.
    if isinstance(f, Coordinate):
        return f == Coordinate(P) and topdim(P) == ambient_dim(P)
    return not isinstance(f, LEAVES)


def apply_gradients(e):
    # A pull-back over an evaluated gradient is pointwise again, so expand_pull_backs finishes.
    return expand_pull_backs(rewrite(e))


def rewrite(e):
    # Inner gradients first, then the rule for the expression of each gradient.
    match map_fields(e, rewrite):
        case Gradient(f, P) if evaluated(f, P):
            return gradient(f, P)
        case e:
            return e


def in_n6(e):
    match e:
        case Gradient(f, P) if evaluated(f, P):
            return False
    return all(in_n6(c) for c in expressions(e))
