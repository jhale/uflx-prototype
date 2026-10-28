import numpy as np

from uflx_spec_interpreter import RD, component_tensor, index, index_sum, pull_back, tensor
from uflx_spec_interpreter._src.domains import cell_of
from uflx_spec_interpreter._src.nodes import (
    ComponentTensor,
    DomainMap,
    Float,
    Function,
    GatheredIndex,
    Gradient,
    Indexed,
    PullBack,
)
from uflx_spec_interpreter._src.passes.pullbacks import expand_pull_backs
from uflx_spec_interpreter._src.substitution import map_fields
from uflx_toy.basix_nodes import BasixCell, BasixSpace
from uflx_toy.nodes import IndexTable
from uflx_toy.runtime import runtime_array

from .evaluation import evaluate
from .quadrature import RULES


def tabulate(f):
    # A function of a Basix space read on a cell at the points of a rule, possibly through the
    # inclusion of a facet, becomes its gathered coefficients times the tabulated basis.
    def rewrite(e):
        e = map_fields(e, rewrite)
        found = leaf(e)
        return values(*found) if found else e

    return rewrite(f)


def coefficients(u):
    return runtime_array(("coefficients", u), (u.space.num_dofs,))


def leaf(e):
    # The function, its cell, the order of the derivative and the points in the reference cell.
    match e:
        case PullBack(inner, DomainMap() as M) if rule_of(M) is not None:
            pass
        case _:
            return None
    points = M.parameterisation
    match inner:
        case PullBack(x, DomainMap(BasixCell(), _, BasixCell()) as A):
            points = expand_pull_backs(pull_back(A.parameterisation, M))
        case x:
            pass
    match x:
        case PullBack(Function(BasixSpace()) as u, C) if cell(C) is not None:
            return u, cell(C), 0, points
        case Gradient(PullBack(Function(BasixSpace()) as u, C), BasixCell()) if cell(C) is not None:
            return u, cell(C), 1, points
    return None


def rule_of(M):
    match M.parameter_domain, M.parameterisation:
        case (RD(0), ComponentTensor(Indexed(P, (_, _)), _)) if P in RULES:
            return P
    return None


def cell(C):
    # The piece and the cell entry of a cell domain or a gathered cell domain.
    j = cell_of(C)
    if j is not None:
        ((c, _),) = C.parameterisation.free_indices
        return j, c
    match C.parameterisation:
        case ComponentTensor(
            Indexed(ComponentTensor(Indexed(psi, _), (_, _)), (GatheredIndex() as g, _)), _
        ):
            pieces = [piece.parameterisation for piece in C.ambient_domain.pieces]
            return pieces.index(psi), g
    return None


def values(u, found_cell, derivative, points):
    # The table has an axis for each free index of the points, such as the point and the facet.
    V, (j, entry) = u.space, found_cell
    axes = [i for i, _ in points.free_indices]
    X = np.real(np.asarray(evaluate(points, {}, np)))
    flat = X.reshape(-1, X.shape[-1])
    tables = V.element(j).tabulate(derivative, flat)[..., 0]
    i = index(("compiler.tabulation", u, entry, points, "i"))
    dofs = IndexTable(V.dofmap[j], V.num_dofs)
    coefficient = coefficients(u)[GatheredIndex(dofs, (entry, i))]
    if derivative == 0:
        phi = table(tables[0].reshape(*X.shape[:-1], -1))
        return index_sum(coefficient * phi[(*axes, i)], (i,))
    d = index(("compiler.tabulation", u, entry, points, "d"))
    dphi = table(np.moveaxis(tables[1:], 0, -1).reshape(*X.shape[:-1], -1, X.shape[-1]))
    return component_tensor(index_sum(coefficient * dphi[(*axes, i, d)], (i,)), (d,))


def table(values):
    values = np.asarray(values)
    if values.ndim == 0:
        return Float(float(values))
    return tensor(tuple(table(v) for v in values))
