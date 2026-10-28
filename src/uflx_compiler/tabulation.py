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
from uflx_spec_interpreter._src.substitution import bottom_up
from uflx_toy.basix_nodes import BasixCell, BasixSpace
from uflx_toy.nodes import IndexTable, RuntimeArray
from uflx_toy.runtime import runtime_array

from .evaluation import evaluate
from .inputs import Basis, runtime_arrays


def tabulate(f):
    # A function of a Basix space read on a cell at the points of a rule, possibly through the
    # inclusion of a facet, becomes its gathered coefficients times the tabulated basis.
    def rewrite(e):
        found = leaf(e)
        return values(*found) if found else e

    return bottom_up(f, rewrite)


def coefficients(u):
    # A label that a consumer writes, from the label of the function.
    return runtime_array(f"coefficients:{u.label}", (u.space.num_dofs,))


def leaf(e):
    # The function, its cell, the order of the derivative and the points in the reference cell.
    match e:
        case PullBack(
            PullBack(x, DomainMap(BasixCell(), _, BasixCell()) as A),
            DomainMap(RD(0), _, BasixCell()) as M,
        ):
            return read(x, expand_pull_backs(pull_back(A.parameterisation, M)))
        case PullBack(x, DomainMap(RD(0), X, BasixCell())):
            return read(x, X)
    return None


def read(x, points):
    match x:
        case PullBack(Function(BasixSpace()) as u, C) if cell(C) is not None:
            return u, cell(C), 0, points
        case Gradient(PullBack(Function(BasixSpace()) as u, C), BasixCell()) if cell(C) is not None:
            return u, cell(C), 1, points
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
    # The gathered coefficients of the cell times the table, summed over the dofs of the cell.
    V, (j, entry) = u.space, found_cell
    axes = [i for i, _ in points.free_indices]
    i, d = (index(("compiler.tabulation", u, entry, points, n)) for n in "id")
    ds = (d,) if derivative else ()
    dofs = IndexTable(V.dofmap[j], V.num_dofs)
    coefficient = coefficients(u)[GatheredIndex(dofs, (entry, i))]
    phi = basis_table(V, j, derivative, points)
    value = index_sum(coefficient * phi[(*axes, i, *ds)], (i,))
    return component_tensor(value, ds) if ds else value


def basis_table(V, j, derivative, points):
    # An axis for each free index of the points, such as the point and the facet, then the dof
    # and the derivative. Points known only at assembly give a table that runtime_tables fills.
    E = V.element(j)
    if not runtime_arrays(points):
        return table(basis(E, derivative, np.real(np.asarray(evaluate(points, {}, np)))))
    extents = [m for _, m in points.free_indices]
    derivatives = (points.value_shape[0],) if derivative else ()
    return runtime_array(Basis(V, j, points, derivative), (*extents, E.dim, *derivatives))


def basis(E, derivative, X):
    # The basis, or its reference gradient, at points of any leading shape.
    tables = E.tabulate(derivative, X.reshape(-1, X.shape[-1]))[..., 0]
    if derivative == 0:
        return tables[0].reshape(*X.shape[:-1], -1)
    return np.moveaxis(tables[1:], 0, -1).reshape(*X.shape[:-1], -1, X.shape[-1])


def runtime_tables(e, env):
    # The library tabulates at the points that the solver gives, before the kernel runs.
    found = {}
    for n in runtime_arrays(e):
        match n:
            case RuntimeArray(Basis(V, j, points, derivative) as label, _):
                X = np.real(np.asarray(evaluate(points, env, np)))
                found[label] = basis(V.element(j), derivative, X)
    return env | found


def table(values):
    values = np.asarray(values)
    if values.ndim == 0:
        return Float(float(values))
    return tensor(tuple(table(v) for v in values))
