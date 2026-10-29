from ..arithmetic import divide, square_root
from ..domains import cell_of, inclusion, is_cell
from ..geometry import inverse_jacobian
from ..nodes import (
    AbstractExpression,
    ComponentTensor,
    Coordinate,
    DomainMap,
    Function,
    Gradient,
    Index,
    Indexed,
    IndexSum,
    Normal,
    Product,
    PullBack,
    SideMap,
    Sum,
)
from ..static import free, shape, support, zero_like
from ..substitution import bottom_up, everywhere, freshen, map_fields, memoized
from ..tensors import transpose


def rule(pb, f, m):
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
            B, A, _ = side_maps(sm, m)
            return pb(pb(g, B), A)
        case PullBack(_, phi) if kept(phi):
            return PullBack(f, m)
        case PullBack(g, phi):
            return pb(g, composed(phi, m, pb))
        case Gradient(g, _) if is_cell(m):
            return contract_gradient(
                Gradient(pb(g, m), m.parameter_domain), inverse_jacobian(m), f, m
            )
        case Gradient():
            return PullBack(f, m)
        case Normal(sm) if cell_of(m) is not None and m.ambient_domain == sm.parameter_domain:
            return normal(f, m, pb)
        case Function() | Normal() | Coordinate():
            return PullBack(f, m)
        case _:
            return pointwise(f, m, pb)


def kept(m):
    # A pull-back stays inside another only by a cell or a side map.
    return isinstance(m, SideMap) or is_cell(m)


def composed(phi, m, pb):
    # The map phi after m, so that only a pull-back by a cell keeps a pull-back inside it.
    return DomainMap(m.parameter_domain, pb(phi.parameterisation, m), phi.ambient_domain)


def broadcast(f, m):
    # f has no support, so the pull-back is f on every member of the family of m.
    pulled = PullBack(f, m)
    if free(pulled).keys() <= free(f).keys():
        return f
    return Sum(f, zero_like(pulled, f.argument_spaces))


def pointwise(f, m, pb):
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
    # B and A of the group of the cells m, and the reference normal of the side tables.
    S, i = sm.parameter_domain.topology, cell_of(m)
    B, A = inclusion(sm.ambient_domain, S, i, sm.side)
    return B, A, S.side_tables(i, sm.side)[2]


def normal(f, m, pb):
    # K^T m_hat / |K^T m_hat|, with m_hat the reference normal of the side tables.
    B, A, m_hat = side_maps(f.side_map, m)
    w = transpose(pb(inverse_jacobian(B), A)) @ m_hat
    return divide(w, square_root(w @ w))


def expand_pull_backs(f):
    # Inner pull-backs first, then the rule for the expression of each pull-back.
    pb = memoized(rule)

    def expand(e):
        match e:
            case PullBack(g, m):
                return pb(g, m)
        return e

    return bottom_up(f, expand)


def in_n5(e):
    def holds(n):
        match n:
            case PullBack(_, SideMap()) | PullBack(Function() | Normal() | Coordinate(), _):
                return True
            case PullBack(PullBack(_, phi), _):
                return kept(phi)
            case PullBack(Gradient(), m):
                return not is_cell(m)
        return not isinstance(n, PullBack)

    return everywhere(e, holds)
