from ..algebra import dot, matvec, minus, scale, transpose, unit
from ..domains import (
    CellMap,
    CoordDomain,
    Decomposed,
    DomainMap,
    Gathered,
    SideMap,
    topdim,
)
from ..geometry import inverse_jacobian
from ..nodes import (
    CT,
    Abs,
    Coordinate,
    Div,
    Fn,
    Gradient,
    Index,
    Indexed,
    ISum,
    Normal,
    Num,
    Prod,
    PullBack,
    Space,
    Sum,
    Tensor,
)
from ..static import arity, free, shape, support, zero_like
from ..substitution import freshen, map_fields, operands

# Nodes in the fields of an expression that are not expressions, so a pull-back leaves them.
NOT_EXPRESSIONS = (Index, Gathered, Space, CoordDomain, Decomposed, DomainMap, CellMap, SideMap)


def pb(f, m):
    # The rule table of expand_pull_backs, for PullBack(f, m).
    match f:
        case _ if isinstance(m, SideMap) and support(f) is not None:
            return PullBack(f, m)
        case _ if support(f) is None:
            return broadcast(f, m)
        case Coordinate(Q) if Q == m.Q:
            return m.psi
        case PullBack(g, SideMap() as sm) if isinstance(m, CellMap) and m.D == sm.E:
            B, A = side_maps(sm, m)
            return pb(pb(g, B), A)
        case PullBack():
            return PullBack(f, m)
        case Gradient(_, Q) if topdim(m.P) < topdim(Q):
            return PullBack(f, m)
        case Gradient(g, _):
            return contract_gradient(Gradient(pb(g, m), m.P), inverse_jacobian(m), f, m)
        case Normal() if isinstance(m, CellMap) and m.D == f.E:
            return normal(f, m)
        case Fn() | Normal() | Coordinate():
            return PullBack(f, m)
        case _:
            return pointwise(f, m)


def broadcast(f, m):
    # f has no support, so the pull-back is f on every member of the family of m.
    pulled = PullBack(f, m)
    return f if free(pulled).keys() <= free(f).keys() else Sum(f, zero_like(pulled, arity(f)))


def pointwise(f, m):
    fresh = freshen(f, [m.psi]) if f.scope else f
    return map_fields(fresh, lambda x: x if isinstance(x, NOT_EXPRESSIONS) else pb(x, m))


def fresh(f, m, n):
    return Index(("uflx.expand_pull_backs", f, m, n))


def contract_gradient(gradP, K, f, m):
    # Entry j of the last axis is the sum over r of gradP[..., r] * K[r, j].
    axes = tuple(fresh(f, m, n) for n in range(len(shape(gradP)) - 1))
    r, j = fresh(f, m, "r"), fresh(f, m, "j")
    return CT(ISum(Prod(Indexed(gradP, (*axes, r)), Indexed(K, (r, j))), (r,)), (*axes, j))


def side_maps(sm, m):
    # The cell on the side and the inclusion of the cell of E into it, for the cells of m.
    e = m.entry
    B = CellMap(sm.D, 0, Gathered(sm.cells, (e,)))
    I = sm.E.pieces[0].P
    k = Index(("uflx.inclusion", sm, e))
    p0, p1 = (Indexed(sm.positions, (e, v, k)) for v in range(2))
    psi = CT(Sum(p0, Prod(minus(p1, p0), Indexed(Coordinate(I), (0,)))), (k,))
    return B, DomainMap(I, psi, B.P)


def normal(f, m):
    # Kᵀm̂ / |Kᵀm̂|, with m̂ the normal of the edge in the reference triangle, away from its centroid.
    B, A = side_maps(f.side, m)
    p0, p1 = ([Indexed(f.side.positions, (m.entry, v, k)) for k in range(2)] for v in range(2))
    t = [minus(b, a) for a, b in zip(p0, p1, strict=True)]
    m0 = Tensor((t[1], Prod(Num(-1), t[0])))
    away = dot(m0, Tensor(tuple(minus(a, Num(1 / 3)) for a in p0)))
    mhat = scale(Div(away, Abs(away)), m0)
    K = pb(inverse_jacobian(B), A)
    return unit(matvec(transpose(K), mhat))


def expand_pull_backs(e):
    # Inner pull-backs first, then the rule for the expression of each pull-back.
    match map_fields(e, expand_pull_backs):
        case PullBack(f, m):
            return pb(f, m)
        case e:
            return e


def in_n5(e):
    match e:
        case PullBack(_, SideMap()) | PullBack(Fn() | Normal() | Coordinate() | PullBack(), _):
            pass
        case PullBack(Gradient(_, Q), m) if topdim(m.P) < topdim(Q):
            pass
        case PullBack():
            return False
    return all(in_n5(c) for c in operands(e))
