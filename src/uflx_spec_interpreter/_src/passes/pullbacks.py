from ..arithmetic import divide, square_root
from ..calculus import rotation_of_gradient, trace_of_gradient
from ..domains import cell_of, inclusion, is_cell
from ..geometry import density, inverse_jacobian, jacobian
from ..names import derived_index
from ..nodes import (
    AbstractExpression,
    Action,
    CellMap,
    ComponentTensor,
    Coordinate,
    Curl,
    Divergence,
    Division,
    DomainMap,
    Form,
    Gradient,
    Indexed,
    IndexSum,
    Normal,
    Product,
    PullBack,
    PushForward,
    SideMap,
    Sum,
    TransformationLaw,
)
from ..static import free, shape, support, terminal, zero_like
from ..substitution import bottom_up, everywhere, freshen, map_fields, memoized, substitute
from ..tensors import determinant, transpose
from ..topology import cell_index
from .actions import apply_actions


def rule(pb, f, m):
    # The rule table of expand_pull_backs, for PullBack(f, m).
    match f:
        case Action(A, p, g):
            return apply_actions(Action(pb(A, m), p, g))
        case _ if isinstance(m, SideMap) and support(f) is not None:
            return PullBack(f, m)
        case _ if support(f) is None:
            return broadcast(f, m)
        case Coordinate():
            return m.parameterisation
        case PullBack(g, SideMap() as sm) if side_cell(sm, m) is not None:
            B, A, _ = side_maps(sm, *side_cell(sm, m))
            return pb(pb(g, B), A)
        case PullBack(_, phi) if kept(phi):
            return PullBack(f, m)
        case PullBack(g, phi):
            return pb(g, composed(phi, m, pb))
        case Gradient(g, _) if is_cell(m):
            return transformed(
                Gradient(pb(g, m), m.parameter_domain),
                TransformationLaw(("covariant",), 0),
                m,
            )
        case Gradient():
            return PullBack(f, m)
        case Divergence(PushForward(g, D, TransformationLaw(("contravariant",), 1)), P) if (
            P == D == m.ambient_domain and is_cell(m)
        ):
            # The divergence on the reference cell, a density, with no derivative of J.
            div = trace_of_gradient(on_the_reference_cell(g, m, pb), m.parameter_domain)
            return transformed(div, TransformationLaw((), 1), m)
        case Curl(PushForward(g, D, TransformationLaw(("covariant",), 0)), P) if (
            P == D == m.ambient_domain and is_cell(m) and k_is_n(m)
        ):
            # The curl on the reference cell, contravariant with the weight 1, with no derivative
            # of J. The identity needs cells of the dimension of the mesh.
            rot = rotation_of_gradient(on_the_reference_cell(g, m, pb), m.parameter_domain)
            return transformed(
                rot, TransformationLaw(("contravariant",) * len(f.value_shape), 1), m
            )
        case Divergence(g, P):
            return pb(trace_of_gradient(g, P), m)
        case Curl(g, P):
            return pb(rotation_of_gradient(g, P), m)
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
    # The parameterisation of a map that a row builds is lowered as the row builds it, because
    # a lowering pass does not enter maps.
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
    if isinstance(fresh, Form):
        return Form(pb(fresh.expression, m), fresh.arguments)
    return map_fields(fresh, lambda x: pb(x, m) if isinstance(x, AbstractExpression) else x)


def derived_axis(f, m, axis):
    """Identify a pull-back axis using the shared structural naming rule."""
    return derived_index("uflx.expand_pull_backs", f, m, axis)


def pushed_forward(g, law, C, pb):
    # The law applied to the value of g on the cell C.
    return transformed(on_the_reference_cell(g, C, pb), law, C)


def on_the_reference_cell(g, C, pb):
    # The value of g on the topology, read on the cell C through its cell map.
    j, e = cell_of(C)
    return pb(g, CellMap(C.ambient_domain.topology, j, e))


def k_is_n(C):
    return C.parameter_domain.dimension == C.ambient_domain.ambient_coordinate_domain.dimension


def transformed(ghat, law, C):
    # K^T on each covariant axis and J on each contravariant axis of ghat, divided by det J, or
    # by the density on a cell of lower dimension, to the weight.
    J, K = jacobian(C), inverse_jacobian(C)
    r, p = len(shape(ghat)), len(law.variances)
    lead = tuple(derived_axis(ghat, C, ("lead", n)) for n in range(r - p))
    out = tuple(derived_axis(ghat, C, ("out", n)) for n in range(p))
    summed = tuple(derived_axis(ghat, C, ("sum", n)) for n in range(p))
    value = Indexed(ghat, (*lead, *summed))
    for variance, i, s in zip(law.variances, out, summed, strict=True):
        M = Indexed(K, (s, i)) if variance == "covariant" else Indexed(J, (i, s))
        value = Product(value, M)
    if summed:
        value = IndexSum(value, summed)
    k, n = C.parameter_domain.dimension, C.ambient_domain.ambient_coordinate_domain.dimension
    scale = determinant(jacobian(C)) if k == n else density(C)
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


def remains(f, m):
    # The pull-backs that no row of expand_pull_backs rewrites, which a library reads.
    return not isinstance(f, Action) and (
        isinstance(m, SideMap)
        or terminal(f)
        or isinstance(f, PushForward)
        or (isinstance(f, PullBack) and kept(f.domain))
        or (isinstance(f, Gradient) and not is_cell(m))
    )


def in_n5(e):
    return everywhere(e, lambda n: not isinstance(n, PullBack) or remains(n.expression, n.domain))
