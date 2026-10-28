from ..arithmetic import subtract, total
from ..geometry import inverse_jacobian
from ..nodes import (
    Absolute,
    AbstractExpression,
    CellMap,
    ComponentTensor,
    Coordinate,
    Division,
    DomainMap,
    Function,
    GatheredIndex,
    Gradient,
    Index,
    Indexed,
    IndexSum,
    Integer,
    Normal,
    Product,
    PullBack,
    SideMap,
    Sum,
    Tensor,
)
from ..static import free, shape, support, zero_like
from ..substitution import expressions, freshen, map_fields
from ..tensors import transpose, unit


def pb(f, m):
    # The rule table of expand_pull_backs, for PullBack(f, m).
    match f:
        case _ if isinstance(m, SideMap) and support(f) is not None:
            return PullBack(f, m)
        case _ if support(f) is None:
            return broadcast(f, m)
        case Coordinate(Q) if Q == m.ambient_domain:
            return m.parameterisation
        case PullBack(g, SideMap() as sm) if (
            isinstance(m, CellMap) and m.ambient_domain == sm.parameter_domain
        ):
            B, A = side_maps(sm, m)
            return pb(pb(g, B), A)
        case PullBack():
            return PullBack(f, m)
        case Gradient(_, Q) if m.parameter_domain.dimension < Q.dimension:
            return PullBack(f, m)
        case Gradient(g, _):
            return contract_gradient(
                Gradient(pb(g, m), m.parameter_domain), inverse_jacobian(m), f, m
            )
        case Normal() if isinstance(m, CellMap) and m.ambient_domain == f.parameter_domain:
            return normal(f, m)
        case Function() | Normal() | Coordinate():
            return PullBack(f, m)
        case _:
            return pointwise(f, m)


def broadcast(f, m):
    # f has no support, so the pull-back is f on every member of the family of m.
    pulled = PullBack(f, m)
    if free(pulled).keys() <= free(f).keys():
        return f
    return Sum(f, zero_like(pulled, f.argument_spaces))


def pointwise(f, m):
    fresh = freshen(f, [m.parameterisation]) if f.scope else f
    return map_fields(fresh, lambda x: pb(x, m) if isinstance(x, AbstractExpression) else x)


def fresh(f, m, n):
    return Index(("uflx.expand_pull_backs", f, m, n))


def contract_gradient(gradP, K, f, m):
    # Entry j of the last axis is the sum over r of gradP[..., r] * K[r, j].
    axes = tuple(fresh(f, m, n) for n in range(len(shape(gradP)) - 1))
    r, j = fresh(f, m, "r"), fresh(f, m, "j")
    return ComponentTensor(
        IndexSum(Product(Indexed(gradP, (*axes, r)), Indexed(K, (r, j))), (r,)), (*axes, j)
    )


def side_maps(sm, m):
    # The cell on the side and the inclusion of the cell of E into it, for the cells of m.
    e = m.cell_index
    B = CellMap(sm.ambient_domain, 0, GatheredIndex(sm.cells, (e,)))
    I = sm.parameter_domain.pieces[0].parameter_domain
    k = Index(("uflx.inclusion", sm, e))
    # The affine map through the positions of the vertices of the cell of E, a constant for a point.
    p = [Indexed(sm.positions, (e, v, k)) for v in range(I.dimension + 1)]
    X = Coordinate(I)
    edges = [Product(subtract(p[v], p[0]), Indexed(X, (v - 1,))) for v in range(1, len(p))]
    return B, DomainMap(I, ComponentTensor(total([p[0], *edges]), (k,)), B.parameter_domain)


def normal(f, m):
    # Kᵀm̂ / |Kᵀm̂|, with m̂ the normal of the edge in the reference triangle, away from its centroid.
    B, A = side_maps(f.side, m)
    p0, p1 = ([Indexed(f.side.positions, (m.cell_index, v, k)) for k in range(2)] for v in range(2))
    t = [subtract(b, a) for a, b in zip(p0, p1, strict=True)]
    m0 = Tensor((t[1], Product(Integer(-1), t[0])))
    away = m0 @ Tensor(tuple(subtract(a, Division(Integer(1), Integer(3))) for a in p0))
    mhat = Division(away, Absolute(away)) * m0
    K = pb(inverse_jacobian(B), A)
    return unit(transpose(K) @ mhat)


def expand_pull_backs(f):
    # Inner pull-backs first, then the rule for the expression of each pull-back.
    match map_fields(f, expand_pull_backs):
        case PullBack(g, m):
            return pb(g, m)
        case e:
            return e


def in_n5(e):
    match e:
        case PullBack(_, SideMap()) | PullBack(
            Function() | Normal() | Coordinate() | PullBack(), _
        ):
            pass
        case PullBack(Gradient(_, Q), m) if m.parameter_domain.dimension < Q.dimension:
            pass
        case PullBack():
            return False
    return all(in_n5(c) for c in expressions(e))
