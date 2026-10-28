import numpy as np

from uflx_spec_interpreter import component_tensor, index, index_sum, tensor
from uflx_spec_interpreter._src.domains import cell_of
from uflx_spec_interpreter._src.nodes import (
    ComponentTensor,
    Float,
    Function,
    GatheredIndex,
    Gradient,
    Indexed,
    PullBack,
)
from uflx_spec_interpreter._src.substitution import map_fields
from uflx_toy.basix_nodes import BasixSpace
from uflx_toy.nodes import IndexTable
from uflx_toy.runtime import runtime_array

from .quadrature import RULES


def tabulate(f):
    # A function of a Basix space at the points of a rule in a cell becomes a sum over the
    # dofs of the cell of the coefficients times the tabulated basis.
    def rewrite(e):
        match map_fields(e, rewrite):
            case PullBack(PullBack(Function(BasixSpace()) as u, cells), points) if (
                cell_of(cells) is not None
            ):
                return values(u, cells, points, 0)
            case PullBack(Gradient(PullBack(Function(BasixSpace()) as u, cells), _), points):
                return values(u, cells, points, 1)
            case e:
                return e

    return rewrite(f)


def coefficients(u):
    return runtime_array(("coefficients", u), (u.space.num_dofs,))


def values(u, cells, points, derivative):
    V, j = u.space, cell_of(cells)
    ((c, _),) = cells.parameterisation.free_indices
    match points.parameterisation:
        case ComponentTensor(Indexed(P, (q, _)), _):
            X = RULES[P]
    i = index(("compiler.tabulation", u, cells, points, "i"))
    dofs = IndexTable(V.dofmap[j], V.num_dofs)
    coefficient = coefficients(u)[GatheredIndex(dofs, (c, i))]
    tables = V.element(j).tabulate(derivative, np.asarray(X, dtype=float))[..., 0]
    if derivative == 0:
        return index_sum(coefficient * table(tables[0])[q, i], (i,))
    d = index(("compiler.tabulation", u, cells, points, "d"))
    gradient = table(np.moveaxis(tables[1:], 0, -1))
    return component_tensor(index_sum(coefficient * gradient[q, i, d], (i,)), (d,))


def table(values):
    values = np.asarray(values)
    if values.ndim == 0:
        return Float(float(values))
    return tensor(tuple(table(v) for v in values))
