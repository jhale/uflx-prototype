from core import Space, CellMap, Gathered, Normal, SideMap, Gateaux, Coordinate, Gradient, Integral, PullBack, ambient_dim, Inv, CT, D, Abs, Action, Conj, Div, Fn, Form, Im, Indexed, ISum, Log, Num, Power, Prod, Re, Sqrt, Sum, Tensor, Zero, operands


def shape(e):
    match e:
        case Gateaux(F, _, _, _):
            return shape(F)
        case Coordinate(P):
            return (ambient_dim(P),)
        case Normal(Dn, _, _):
            return (ambient_dim(Dn),)
        case Gradient(b, P):
            return shape(b) + (ambient_dim(P),)
        case PullBack(b, _) | Integral(b, _):
            return shape(b)
        case Fn(space, _):
            return () if space.dual else space.shape
        case Num() | Prod() | Div() | Indexed() | ISum() | Re() | Im() | Abs() | Sqrt() | Log() | Power():
            return ()
        case Zero(s, _, _):
            return s
        case Inv(a):
            return tuple(reversed(shape(a)))
        case Tensor(es):
            return (len(es),) + shape(es[0])
        case CT(x, idx):
            return tuple(free(x)[i] for i in idx)
        case Sum(a, _) | Conj(a) | D(a, _, _, _) | Action(a, _, _) | Form(a, _):
            return shape(a)
    raise TypeError(e)


def free(e):
    # The free indices with their extents, in order of first occurrence.
    match e:
        case Gateaux(F, _, a, _):
            return free(F) | free(a)
        case Coordinate() | Normal():
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
    for x in operands(e):
        if not isinstance(x, Fn):
            out |= free(x)
    return out


def arity(e):
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
        case Sum(a, _) | Div(a, _) | Conj(a) | Indexed(a, _) | CT(a, _) | ISum(a, _) | D(a, _, _, _):
            return arity(a)
    raise TypeError(e)


def support(e):
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
    for x in operands(e):
        s = support(x)
        if s is not None:
            found = s
    return found


def by_identity(fn):
    # Pass outputs share subexpressions, so a walk without memoisation is exponential in depth.
    seen = {}

    def wrapped(e):
        if id(e) not in seen:
            seen[id(e)] = (e, fn(e))
        value = seen[id(e)][1]
        return dict(value) if isinstance(value, dict) else value

    return wrapped


shape, free, arity = by_identity(shape), by_identity(free), by_identity(arity)


def free_indices(*es):
    merged = {}
    for e in es:
        merged |= free(e)
    return tuple(merged.items())


def zero_like(f, n_args, *others):
    # Placeholder argument spaces, since the toy does not track the spaces of arguments.
    return Zero(shape(f), free_indices(f, *others), tuple(Space("arg") for _ in range(n_args)))
