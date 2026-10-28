from ..arithmetic import divide, subtract
from ..cells import affine_jacobian, affine_map, centroid
from ..domains import cell_of, gathered_cell_domain
from ..geometry import inverse_jacobian
from ..nodes import (
    Absolute,
    AbstractExpression,
    ComponentTensor,
    Coordinate,
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
from ..tensors import component, determinant, transpose, unit


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
            cell_of(m) is not None and m.ambient_domain == sm.parameter_domain
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
        case Normal() if cell_of(m) is not None and m.ambient_domain == f.parameter_domain:
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


def cell_index_of(m):
    # The one free index of the cells of m, a cell domain.
    ((e, _),) = m.parameterisation.free_indices
    return e


def side_maps(sm, m):
    # The cell on the side and the inclusion of the cell of E into it, for the cells of m.
    e = cell_index_of(m)
    B = gathered_cell_domain(sm.ambient_domain, 0, GatheredIndex(sm.cells, (e,)))
    F = m.parameter_domain
    return B, DomainMap(F, affine_map(F, side_points(sm, e)), B.parameter_domain)


def side_points(sm, e):
    # The position in the cell on the side of each vertex v of the cell e.
    v, k = (Index(("uflx.inclusion", sm, e, n)) for n in range(2))
    return ComponentTensor(Indexed(sm.positions, (e, v, k)), (v, k))


def normal(f, m):
    # K^T m / |K^T m|, with m orthogonal to the cell of E in K_j and away from its centroid.
    B, A = side_maps(f.side, m)
    K_j, F = B.parameter_domain, m.parameter_domain
    points = side_points(f.side, cell_index_of(m))
    m0 = cofactor_normal(affine_jacobian(F, points), K_j.dimension)
    away = m0 @ subtract(component(points, (0,)), centroid(K_j))
    mhat = divide(away, Absolute(away)) * m0
    K = pb(inverse_jacobian(B), A)
    return unit(transpose(K) @ mhat)


def cofactor_normal(J, n):
    # Entry i is det of [e_i, the columns of J], orthogonal to the n - 1 columns of J.
    def column_matrix(i):
        rows = (
            Tensor((Integer(1 if r == i else 0), *(Indexed(J, (r, c)) for c in range(n - 1))))
            for r in range(n)
        )
        return Tensor(tuple(rows))

    return Tensor(tuple(determinant(column_matrix(i)) for i in range(n)))


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
