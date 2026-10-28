from .core import (
    CT, Abs, Conj, Coordinate, CoordDomain, CellMap, Decomposed, Div, DomainMap, Fn, Gathered, Gradient,
    Index, Indexed, Integral, Inv, ISum, Log, Normal, Num, Power, Prod, PullBack, Re, Im, SideMap, Sqrt,
    Sum, Tensor, Zero, ambient_dim, topdim, freshen, map_fields, operands, Space,
)
from .static import arity, free, shape, support

def fresh(f, m, n):
    return Index(("uflx.expand_pull_backs", f, m, n))


def contract_gradient(gradP, K, f, m):
    # Entry j of the last axis is the sum over r of gradP[..., r] * K[r, j].
    axes = tuple(fresh(f, m, n) for n in range(len(shape(gradP)) - 1))
    r, j = fresh(f, m, "r"), fresh(f, m, "j")
    return CT(ISum(Prod(Indexed(gradP, axes + (r,)), Indexed(K, (r, j))), (r,)), axes + (j,))


def side_maps(sm, m):
    # The cell on the side and the inclusion of the cell of E into it, for the cells selected by m.
    e = m.entry
    B = CellMap(sm.D, 0, Gathered(sm.cells, (e,)))
    I = sm.E.pieces[0].P
    k = Index(("uflx.inclusion", sm, e))
    pos = lambda v: Indexed(sm.positions, (e, v, k))
    psi = CT(Sum(pos(0), Prod(Sum(pos(1), Prod(Num(-1), pos(0))), Indexed(Coordinate(I), (0,)))), (k,))
    A = DomainMap(I, psi, B.P)
    return B, A


def normal(f, m):
    sm = f.side
    B, A = side_maps(sm, m)
    e = m.entry
    pos = lambda v, k: Indexed(sm.positions, (e, v, k))
    t = [Sum(pos(1, k), Prod(Num(-1), pos(0, k))) for k in range(2)]
    m0 = Tensor((t[1], Prod(Num(-1), t[0])))
    d = Tensor(tuple(Sum(pos(0, k), Num(-1 / 3)) for k in range(2)))
    r = fresh(f, m, "r")
    dot = ISum(Prod(Indexed(m0, (r,)), Indexed(d, (r,))), (r,))
    sign = Div(dot, Abs(dot))
    j, s = fresh(f, m, "j"), fresh(f, m, "s")
    mhat = CT(Prod(sign, Indexed(m0, (s,))), (s,))
    K = Inv(PullBack(Gradient(B.psi, B.P), A))
    v = CT(ISum(Prod(Indexed(K, (r, j)), Indexed(mhat, (r,))), (r,)), (j,))
    q = fresh(f, m, "q")
    length = Sqrt(ISum(Prod(Indexed(v, (q,)), Indexed(v, (q,))), (q,)))
    return CT(Div(Indexed(v, (j,)), length), (j,))


def pb(f, m):
    # The rows of expand_pull_backs, for PullBack(f, m).
    if isinstance(m, SideMap) and support(f) is not None:
        # A restriction waits for the pull-back by a cell of E, which selects the cell on the side.
        return PullBack(f, m)
    if support(f) is None:
        missing = {i: n for i, n in free(PullBack(f, m)).items() if i not in free(f)}
        if not missing:
            return f
        return Sum(f, Zero(shape(f), tuple(missing.items()), tuple(Space("arg") for _ in range(arity(f)))))
    match f:
        case Coordinate(Q) if Q == m.Q:
            return m.psi
        case PullBack(g, SideMap() as sm) if isinstance(m, CellMap) and m.D == sm.E:
            B, A = side_maps(sm, m)
            return pb(pb(g, B), A)
        case PullBack():
            return PullBack(f, m)
        case Gradient(g, Q) if topdim(m.P) < topdim(Q):
            return PullBack(f, m)
        case Gradient(g, Q) if topdim(m.P) == ambient_dim(m.P):
            K = Inv(Gradient(m.psi, m.P))
            return contract_gradient(Gradient(pb(g, m), m.P), K, f, m)
        case Gradient(g, Q):
            # A parameter domain of lower dimension than its ambient: K = (J^T J + I - Pi)^-1 J^T.
            from .algebra import tangent_gram
            J = Gradient(m.psi, m.P)
            a, b, c, l = (fresh(f, m, ("K", n)) for n in range(4))
            G = tangent_gram(J, m.P)
            K = CT(ISum(Prod(Indexed(Inv(G), (a, c)), Indexed(J, (l, c))), (c,)), (a, l))
            return contract_gradient(Gradient(pb(g, m), m.P), K, f, m)
        case Normal() if isinstance(m, CellMap) and m.D == f.E:
            return normal(f, m)
        case Fn() | Normal():
            return PullBack(f, m)
        case Indexed(x, idx):
            return Indexed(pb(x, m), idx)
        case CT() | ISum():
            fr = freshen(f, [m.psi])
            return type(fr)(pb(fr.e, m), fr.idx)
        case Tensor(es):
            return Tensor(tuple(pb(x, m) for x in es))
        case Sum() | Prod() | Div() | Conj() | Re() | Im() | Abs() | Sqrt() | Log() | Power() | Inv():
            return map_fields(f, lambda x: pb(x, m))
    raise TypeError(f"no row: {type(f).__name__}")


def expand_pull_backs(e):
    e = map_fields(e, expand_pull_backs)
    return pb(e.b, e.m) if isinstance(e, PullBack) else e


def in_n5(e):
    if isinstance(e, PullBack):
        b, m = e.b, e.m
        ok = isinstance(b, (Fn, Normal, PullBack)) or (isinstance(b, Gradient) and topdim(m.P) < topdim(b.P))
        if not ok:
            return False
    return all(in_n5(c) for c in operands(e))
