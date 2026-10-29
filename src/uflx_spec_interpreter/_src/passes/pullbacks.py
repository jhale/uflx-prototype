from ..arithmetic import divide, square_root
from ..domains import cell_of, inclusion, is_cell
from ..geometry import density, inverse_jacobian, jacobian
from ..nodes import (
    AbstractExpression,
    CellMap,
    ComponentTensor,
    Coordinate,
    Division,
    DomainMap,
    Gradient,
    Index,
    Indexed,
    IndexSum,
    Normal,
    Product,
    PullBack,
    PushForward,
    SideMap,
    Sum,
)
from ..static import free, shape, support, terminal, zero_like
from ..substitution import bottom_up, everywhere, freshen, map_fields, memoized, substitute
from ..tensors import determinant, transpose
from ..topology import cell_index


def rule(pb, f, m):
    # The rule table of expand_pull_backs, for PullBack(f, m).
    match f:
        case _ if isinstance(m, SideMap) and support(f) is not None:
            return PullBack(f, m)
        case _ if support(f) is None:
            return broadcast(f, m)
        case Coordinate():
            return lowered_copy(pb, m.parameterisation)
        case PullBack(g, SideMap() as sm) if side_cell(sm, m) is not None:
            B, A, _ = side_maps(sm, *side_cell(sm, m))
            return pb(pb(g, B), A)
        case PullBack(_, phi) if kept(phi):
            return PullBack(f, m)
        case PullBack(g, phi):
            return pb(g, composed(phi, m, pb))
        case Gradient(g, _) if is_cell(m):
            K = lowered_copy(pb, inverse_jacobian(m))
            return contract_gradient(Gradient(pb(g, m), m.parameter_domain), K, f, m)
        case Gradient():
            return PullBack(f, m)
        case Normal(sm) if side_cell(sm, m) is not None:
            return normal(f, m, pb)
        case PushForward(g, D, law) if m.ambient_domain == D and is_cell(m):
            return pushed_forward(g, law, m, pb)
        case _ if terminal(f) or isinstance(f, PushForward):
            return PullBack(f, m)
        case _:
            return pointwise(f, m, pb)


def kept(m):
    # A pull-back stays inside another only by a cell, a cell map or a side map.
    return isinstance(m, SideMap | CellMap) or is_cell(m)


def composed(phi, m, pb):
    # The map phi after m, so that only a pull-back by a cell keeps a pull-back inside it.
    psi = pb(lowered_copy(pb, phi.parameterisation), m)
    return DomainMap(m.parameter_domain, psi, phi.ambient_domain)


def lowered_copy(pb, e):
    # An expression read out of a domain map, which a lowering pass does not enter, is lowered
    # where the rows bring it into the expression.
    return bottom_up(e, lambda n: pb(n.expression, n.domain) if isinstance(n, PullBack) else n)


def broadcast(f, m):
    # f has no support, so the pull-back is f on every member of the family of m.
    pulled = PullBack(f, m)
    if free(pulled).keys() <= free(f).keys():
        return f
    return Sum(f, zero_like(pulled, f.argument_spaces))


def pointwise(f, m, pb):
    # A binding node is made fresh for the free indices of the map, which occur in the map.
    fresh = freshen(f, [m])
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


def pushed_forward(g, law, C, pb):
    # The law applied to the value of g on the cell C, read through its cell map: K^T on each
    # covariant axis and J on each contravariant axis, divided by det J, or by the density on a
    # cell of lower dimension, to the weight.
    j, e = cell_of(C)
    ghat = pb(g, CellMap(C.ambient_domain.topology, j, e))
    J, K = (lowered_copy(pb, M(C)) for M in (jacobian, inverse_jacobian))
    r, p = len(shape(ghat)), len(law.kinds)
    lead = tuple(fresh(g, C, ("lead", n)) for n in range(r - p))
    out = tuple(fresh(g, C, ("out", n)) for n in range(p))
    summed = tuple(fresh(g, C, ("sum", n)) for n in range(p))
    value = Indexed(ghat, (*lead, *summed))
    for kind, i, s in zip(law.kinds, out, summed, strict=True):
        value = Product(value, Indexed(K, (s, i)) if kind == "covariant" else Indexed(J, (i, s)))
    if summed:
        value = IndexSum(value, summed)
    k, n = C.parameter_domain.dimension, C.ambient_domain.ambient_coordinate_domain.dimension
    scale = lowered_copy(pb, determinant(jacobian(C)) if k == n else density(C))
    for _ in range(law.weight):
        value = Division(value, scale)
    axes = (*lead, *out)
    return ComponentTensor(value, axes) if axes else value


def side_cell(sm, m):
    # The group and the entry of m as a cell of the parameter domain of the side map: a cell
    # map of the side topology, or a cell of the side domain.
    if isinstance(m, CellMap):
        return (m.group, m.entry) if m.topology == sm.side_topology else None
    return cell_of(m) if m.ambient_domain == sm.parameter_domain else None


def side_maps(sm, i, e):
    # B and A of cell e of group i, and the reference normal of the side tables.
    S, s = sm.side_topology, sm.sides[i]
    B, A = inclusion(sm.ambient_domain, S, i, s, e)
    m_hat = S.side_tables(i, s)[2]
    return B, A, None if m_hat is None else substitute(m_hat, {cell_index(S, i): e})


def normal(f, m, pb):
    # K^T m_hat / |K^T m_hat|, with m_hat the reference normal of the side tables.
    B, A, m_hat = side_maps(f.side_map, *side_cell(f.side_map, m))
    w = transpose(pb(lowered_copy(pb, inverse_jacobian(B)), A)) @ m_hat
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


def remains(f, m):
    # The pull-backs that no row of expand_pull_backs rewrites, which a library reads.
    return (
        isinstance(m, SideMap)
        or terminal(f)
        or isinstance(f, PushForward)
        or (isinstance(f, PullBack) and kept(f.domain))
        or (isinstance(f, Gradient) and not is_cell(m))
    )


def in_n5(e):
    return everywhere(e, lambda n: not isinstance(n, PullBack) or remains(n.expression, n.domain))
