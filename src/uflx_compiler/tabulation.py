import basix
import numpy as np

from uflx_spec_interpreter import (
    RD,
    index_sum,
    tensor,
)
from uflx_spec_interpreter._src.algorithms.names import _derived_index, _derived_indices
from uflx_spec_interpreter._src.algorithms.traversal import _bottom_up
from uflx_spec_interpreter._src.meshes import _cell_of
from uflx_spec_interpreter._src.nodes import (
    CellMap,
    DomainMap,
    Function,
    GatheredIndex,
    Gradient,
    PullBack,
    Tensor,
    Zero,
)
from uflx_spec_interpreter._src.nodes.scalar_types import ScalarType
from uflx_toy.basix_nodes import BasixCell, BasixSpace
from uflx_toy.basix_spaces import basix_space
from uflx_toy.nodes import IndexTable, RuntimeArray, VertexCoordinates
from uflx_toy.runtime import runtime_array

from .evaluation import evaluate
from .inputs import Basis, runtime_arrays
from .quadrature import constant


def tabulate(f):
    # A function of a Basix space read on a cell at the points of a rule, possibly through the
    # inclusion of a facet, becomes its gathered coefficients times the tabulated basis.
    def rewrite(e):
        # A function with no domain is a constant, which a consumer passes by its label.
        if isinstance(e, Function) and e.domain is None:
            return runtime_array(f"coefficients:{e.label}", e.value_shape)
        found = leaf(e)
        if found and isinstance(found[0], VertexCoordinates):
            return vertex_values(*found)
        return values(*found) if found else e

    return _bottom_up(f, rewrite)


def coefficients(u):
    # A label that a consumer writes, from the label of the function.
    return runtime_array(f"coefficients:{u.label}", (u.space.num_dofs,))


def leaf(e):
    # The function, its cell, the order of the derivative and the points in the reference cell.
    match e:
        case PullBack(x, DomainMap(RD(0), X, BasixCell())):
            return read(x, X)
    return None


def read(x, points):
    # The order of the derivative is the number of reference gradients around the read.
    order = 0
    while isinstance(x, Gradient) and isinstance(x.domain, BasixCell):
        x, order = x.expression, order + 1
    match x:
        case PullBack(VertexCoordinates() | Function(BasixSpace()) as u, C) if cell(C) is not None:
            return u, cell(C), order, points
    return None


def cell(C):
    # The group and the entry of a cell map, or of a cell of a mesh.
    return (C.cell_group, C.entry) if isinstance(C, CellMap) else _cell_of(C)


def values(u, found_cell, order, points):
    # The gathered coefficients of the cell times a table for each factor of the element, summed
    # over a dof index for each factor. An element that does not factor is its one factor.
    V, (j, entry) = u.space, found_cell
    dimension = points.value_shape[0]
    if V.factors(j) and separable(points):
        factors = [(F, X, (k,)) for k, (F, X) in enumerate(zip(V.factors(j), points.entries))]
    else:
        factors = [(V.element(j), points, tuple(range(dimension)))]
    i = _derived_indices("compiler.tabulation", (u, entry, points), len(factors))
    rows = np.asarray(V.dofmap[j]).reshape(-1, *(F.dim for F, _, _ in factors), V.block_size)
    dofs = tuple(IndexTable(nested(rows[..., c]), V.num_dofs) for c in range(V.block_size))

    def term(idx):
        # Component c of Basix and the directions of the derivative, for each factor its own.
        rank = len(V.value_shape)
        c, directions = int(np.ravel_multi_index(idx[:rank], V.value_shape)), idx[rank:]
        out = coefficients(u)[GatheredIndex(dofs[c if V.block_shape else 0], (entry, *i))]
        for k, (F, X, covered) in enumerate(factors):
            own = tuple(covered.index(m) for m in directions if m in covered)
            out = out * table_read(V, j, F, X, own, i[k], 0 if V.block_shape else c)
        return index_sum(out, i)

    return arrayed((*V.value_shape, *(dimension,) * order), term)


def arrayed(shape, entry, idx=()):
    if len(idx) == len(shape):
        return entry(idx)
    return tensor(tuple(arrayed(shape, entry, (*idx, m)) for m in range(shape[len(idx)])))


def vertex_values(x, found_cell, order, points):
    # The coordinates of the vertices of the cell times the shape functions of the vertices, which
    # are the basis of degree 1 of Basix, in the order of the vertices.
    T, (j, entry) = x.topology, found_cell
    # The shape functions are the basis of a space of degree 1 on T, which also labels a table
    # at points known only at assembly.
    V = basix_space("vertices", T, basix.ElementFamily.P, 1)
    F = V.element(j)
    a = _derived_index("compiler.tabulation", x, entry, points, "a")
    # The element may order its dofs for sum factorisation, so each dof reads its own vertex.
    vertex_of = {d: v for v, dofs in enumerate(F.entity_dofs[0]) for d in dofs}
    rows = [[cell[vertex_of[d]] for d in range(F.dim)] for cell in T.cells[j]]
    cells = IndexTable(nested(np.asarray(rows)), len(x.vertices))
    vertex = constant(x.vertices)[GatheredIndex(cells, (entry, a))]

    def term(idx):
        return index_sum(vertex[idx[0]] * table_read(V, j, F, points, idx[1:], a, 0), (a,))

    return arrayed((len(x.vertices[0]), *(points.value_shape[0],) * order), term)


def separable(points):
    # Coordinate k of the points reads indices that no other coordinate reads.
    match points:
        case Tensor(entries) if not runtime_arrays(points):
            seen = [set(dict(x.free_indices)) for x in entries]
            return sum(map(len, seen)) == len(set().union(*seen))
    return False


def nested(values):
    return tuple(map(nested, values)) if isinstance(values, np.ndarray) else int(values)


def table_read(V, j, F, X, d, i, c):
    # A table that does not vary along an axis of the points, such as the derivative of an
    # affine element, is read without that axis, so the read is the same at every point.
    axes = [n for n, _ in X.free_indices]
    if runtime_arrays(X):
        return basis_table(V, j, F, X, d, c)[(*axes, i)]
    values, kept = basis(F, d, coordinates(X, {}), c), []
    if not np.any(values):
        # A table of zeros, such as a second derivative of an affine element, is Zero.
        return Zero((), (*X.free_indices, (i, F.dim)), (), ScalarType.real)
    for a in axes:
        p = len(kept)
        first = np.take(values, [0], axis=p)
        if np.allclose(values, first, rtol=0, atol=1e-13 * max(1.0, np.abs(values).max())):
            values = np.squeeze(first, axis=p)
        else:
            kept.append(a)
    return constant(values)[(*kept, i)]


def basis_table(V, j, F, X, d, c):
    # An axis for each free index of the points X, such as the point and the facet, then the dof.
    # Points known only at assembly give a table that runtime_tables fills.
    if not runtime_arrays(X):
        return constant(basis(F, d, coordinates(X, {}), c))
    extents = [m for _, m in X.free_indices]
    return runtime_array(Basis(V, j, X, d, c), (*extents, F.dim))


def coordinates(X, env):
    # The points as an array with the coordinate last, which a factor reads one at a time.
    values = np.real(np.asarray(evaluate(X, env, np)))
    return values if X.value_shape else values[..., None]


def basis(F, directions, X, c=0):
    # Component c of the basis, or of its reference derivative in the directions, at points of
    # any leading shape.
    counts = [directions.count(m) for m in range(X.shape[-1])]
    tables = F.tabulate(len(directions), X.reshape(-1, X.shape[-1]))[..., c]
    return tables[basix.index(*counts)].reshape(*X.shape[:-1], -1)


def runtime_tables(e, env):
    # The library tabulates at the points that the solver gives, before the kernel runs.
    found = {}
    for n in runtime_arrays(e):
        match n:
            case RuntimeArray(Basis(V, j, points, d, c) as label, _):
                found[label] = basis(V.element(j), d, coordinates(points, env), c)
    return env | found
