from .domains import Gathered, SideMap, ambient_dim
from .nodes import (
    CT,
    Abs,
    Action,
    Conj,
    Coordinate,
    D,
    Div,
    Fn,
    Form,
    Gateaux,
    Gradient,
    Im,
    Indexed,
    Integral,
    Inv,
    ISum,
    Log,
    Normal,
    Num,
    Power,
    Prod,
    PullBack,
    Re,
    Space,
    Sqrt,
    Sum,
    Tensor,
    Zero,
)
from .substitution import expressions


def rule_shape(e):
    match e:
        case Gateaux(F, _, _, _):
            return shape(F)
        case Coordinate(P):
            return (ambient_dim(P),)
        case Normal(Dn, _, _):
            return (ambient_dim(Dn),)
        case Gradient(b, P):
            return (*shape(b), ambient_dim(P))
        case PullBack(b, _) | Integral(b, _):
            return shape(b)
        case Fn(space, _):
            return () if space.dual else space.shape
        case (
            Num()
            | Prod()
            | Div()
            | Indexed()
            | ISum()
            | Re()
            | Im()
            | Abs()
            | Sqrt()
            | Log()
            | Power()
        ):
            return ()
        case Zero(s, _, _):
            return s
        case Inv(a):
            return tuple(reversed(shape(a)))
        case Tensor(es):
            return (len(es), *shape(es[0]))
        case CT(x, idx):
            return tuple(free(x)[i] for i in idx)
        case Sum(a, _) | Conj(a) | D(a, _, _, _) | Action(a, _, _) | Form(a, _):
            return shape(a)
    raise TypeError(e)


def rule_free(e):
    # The free indices with their extents, in order of first occurrence.
    match e:
        case Gateaux(F, _, a, _):
            return free(F) | free(a)
        case Coordinate() | Normal() | Fn():
            return {}
        case Gradient(b, _):
            return free(b)
        case Integral(b, P):
            bound = set()
            for piece in getattr(P, "pieces", ()):
                bound |= set(free(piece.psi))
            return {i: n for i, n in free(b).items() if i not in bound}
        case PullBack(b, m):
            return free(b) | ({} if isinstance(m, SideMap) else free(m.psi))
        case Zero(_, f, _):
            return dict(f)
        case Indexed(x, idx):
            out = free(x)
            for k, i in enumerate(idx):
                if isinstance(i, Gathered):
                    for m, j in enumerate(i.indices):
                        if not isinstance(j, int):
                            out[j] = shape(i.table)[m]
                elif not isinstance(i, int):
                    out[i] = shape(x)[k]
            return out
        case CT(x, idx) | ISum(x, idx):
            return {i: n for i, n in free(x).items() if i not in idx}
        case D(f, _, a, h):
            return free(f) | free(a) | free(h)
    out = {}
    for x in expressions(e):
        out |= free(x)
    return out


def rule_arity(e):
    match e:
        case Gateaux(F, _, _, _):
            return arity(F) + 1
        case Coordinate() | Normal():
            return 0
        case Gradient(b, _) | PullBack(b, _) | Integral(b, _):
            return arity(b)
        case Fn(space, _):
            return 1 if space.dual else 0
        case Zero(_, _, args) | Form(_, args):
            return len(args)
        case Action(f, _, _):
            return arity(f) - 1
        case Prod(a, b):
            return arity(a) + arity(b)
        case Num() | Re() | Im() | Abs() | Sqrt() | Log() | Power() | Inv():
            return 0
        case Tensor(es):
            return arity(es[0])
        case (
            Sum(a, _) | Div(a, _) | Conj(a) | Indexed(a, _) | CT(a, _) | ISum(a, _) | D(a, _, _, _)
        ):
            return arity(a)
    raise TypeError(e)


def rule_support(e):
    # The single support of an expression, or None.
    match e:
        case Coordinate(P):
            return P
        case Fn(space, _):
            return space.domain
        case PullBack(_, m):
            return m.E if isinstance(m, SideMap) else m.P
        case Normal(_, E, _):
            return E
        case Integral() | Num():
            return None
        case Gradient(_, P):
            return P
    found = None
    for x in expressions(e):
        s = support(x)
        if s is not None:
            found = s
    return found


# The free functions read the static type of an expression.
def shape(e):
    return e.value_shape


def free(e):
    return dict(e.free_indices)


def arity(e):
    return e.arity


def support(e):
    return e.support


def free_indices(*es):
    merged = {}
    for e in es:
        merged |= free(e)
    return tuple(merged.items())


def zero_like(f, n_args, *others):
    # Placeholder argument spaces, since the toy does not track the spaces of arguments.
    return Zero(shape(f), free_indices(f, *others), tuple(Space("arg") for _ in range(n_args)))
