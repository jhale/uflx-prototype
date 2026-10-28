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
from uflx_spec_interpreter._src.substitution import expressions, map_fields
from uflx_toy.basix_nodes import BasixCell, BasixSpace
from uflx_toy.nodes import IndexTable, RuntimeArray
from uflx_toy.runtime import runtime_array

from .evaluation import evaluate


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
        case PullBack(inner, DomainMap(RD(0), _, BasixCell()) as M):
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
    # Points known only at assembly give a table that the library computes then.
    V, (j, entry) = u.space, found_cell
    E, axes = V.element(j), [i for i, _ in points.free_indices]
    i = index(("compiler.tabulation", u, entry, points, "i"))
    dofs = IndexTable(V.dofmap[j], V.num_dofs)
    coefficient = coefficients(u)[GatheredIndex(dofs, (entry, i))]
    if reads_input(points):
        extents = [m for _, m in points.free_indices]
        derivatives = (points.value_shape[0],) if derivative else ()
        phi = runtime_array(("basis", V, j, points, derivative), (*extents, E.dim, *derivatives))
    else:
        phi = table(basis(E, derivative, np.real(np.asarray(evaluate(points, {}, np)))))
    if derivative == 0:
        return index_sum(coefficient * phi[(*axes, i)], (i,))
    d = index(("compiler.tabulation", u, entry, points, "d"))
    return component_tensor(index_sum(coefficient * phi[(*axes, i, d)], (i,)), (d,))


def basis(E, derivative, X):
    # The basis, or its reference gradient, at points of any leading shape.
    tables = E.tabulate(derivative, X.reshape(-1, X.shape[-1]))[..., 0]
    if derivative == 0:
        return tables[0].reshape(*X.shape[:-1], -1)
    return np.moveaxis(tables[1:], 0, -1).reshape(*X.shape[:-1], -1, X.shape[-1])


def reads_input(e):
    return isinstance(e, RuntimeArray) or any(reads_input(c) for c in expressions(e))


def runtime_tables(e, env):
    # The library tabulates at the points that the solver gives, before the kernel runs.
    found, seen = {}, set()

    def walk(n):
        if n in seen:
            return
        seen.add(n)
        if isinstance(n, RuntimeArray) and n.label[0] == "basis":
            _, V, j, points, derivative = n.label
            X = np.real(np.asarray(evaluate(points, env, np)))
            found[n.label] = basis(V.element(j), derivative, X)
        for c in expressions(n):
            walk(c)

    walk(e)
    return env | found


def table(values):
    values = np.asarray(values)
    if values.ndim == 0:
        return Float(float(values))
    return tensor(tuple(table(v) for v in values))
