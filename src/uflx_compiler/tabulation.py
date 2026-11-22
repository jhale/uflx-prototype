import basix
import numpy as np

from uflx.algorithms import bottom_up, derived_indices
from uflx.nodes import (
    CellMap,
    DomainMap,
    EuclideanSpace,
    Function,
    GatheredIndex,
    Gradient,
    PullBack,
    ScalarType,
    Tensor,
    Zero,
)
from uflx.tensors import index_sum, tensor
from uflx_basix.basix_nodes import BasixBasis, BasixCell, BasixSpace
from uflx_basix.nodes import IndexTable, RuntimeArray
from uflx_basix.quadrature import constant
from uflx_basix.runtime import runtime_array

from .evaluation import evaluate
from .inputs import Basis, runtime_arrays


def tabulate(f):
    # A function of a Basix space read on a cell at the points of a rule, possibly through the
    # inclusion of a facet, becomes its gathered coefficients times the tabulated basis.
    def rewrite(e):
        # A function with no domain is a constant, which a consumer passes by its label.
        if isinstance(e, Function) and e.domain is None:
            return runtime_array(f"coefficients:{e.label}", e.value_shape)
        found = leaf(e)
        return values(*found) if found else e

    return bottom_up(f, rewrite)


def coefficients(u):
    # A label that a consumer writes, from the label of the function.
    return runtime_array(f"coefficients:{u.label}", (u.space.num_dofs,))


def leaf(e):
    # The function, its cell, the order of the derivative and the points in the reference cell.
    match e:
        case PullBack(x, DomainMap(EuclideanSpace(0), X, BasixCell())):
            return read(x, X)
    return None


def read(x, points):
    # The order of the derivative is the number of reference gradients around the read.
    order = 0
    while isinstance(x, Gradient) and isinstance(x.domain, BasixCell):
        x, order = x.expression, order + 1
    match x:
        case BasixBasis() as u if isinstance(u.domain, BasixCell):
            return u, 0, order, points
        case Function(BasixSpace() as V) as u if isinstance(V.domain, BasixCell):
            return u, 0, order, points
        case PullBack(BasixBasis() | Function(BasixSpace()) as u, C) if cell(C) is not None:
            return u, cell(C), order, points
    return None


def cell(C):
    # The fixed local slot selected by a topology or mesh cell map.
    return C.cell_slot if isinstance(C, CellMap) else None


def values(u, found_cell, order, points):
    if isinstance(u, BasixBasis):
        shape = (*u.value_shape, *(points.value_shape[0],) * order)
        if found_cell != u.cell_slot:
            return Zero(shape, points.free_indices, (), ScalarType.real)
        V, j = u.space, u.cell_slot
        E = V.element(j)
        rank = len(E.value_shape)

        def entry(idx):
            c = int(np.ravel_multi_index(idx[1 : 1 + rank], E.value_shape))
            return table_read(V, j, E, points, idx[1 + rank :], idx[0], c)

        return arrayed(shape, entry)
    # The gathered coefficients of the cell times a table for each factor of the element, summed
    # over a dof index for each factor. An element that does not factor is its one factor.
    V, j = u.space, found_cell
    dimension = points.value_shape[0]
    if V.factors(j) and separable(points):
        factors = [(F, X, (k,)) for k, (F, X) in enumerate(zip(V.factors(j), points.entries))]
    else:
        factors = [(V.element(j), points, tuple(range(dimension)))]
    i = derived_indices("compiler.tabulation", (u, j, points), len(factors))
    rows = np.asarray(V.cell_dofs[j]).reshape(*(F.dim for F, _, _ in factors), V.block_size)
    dofs = tuple(IndexTable(nested(rows[..., c]), V.num_dofs) for c in range(V.block_size))

    def term(idx):
        # Component c of Basix and the directions of the derivative, for each factor its own.
        rank = len(V.value_shape)
        c, directions = int(np.ravel_multi_index(idx[:rank], V.value_shape)), idx[rank:]
        out = coefficients(u)[GatheredIndex(dofs[c if V.block_shape else 0], i)]
        for k, (F, X, covered) in enumerate(factors):
            own = tuple(covered.index(m) for m in directions if m in covered)
            out = out * table_read(V, j, F, X, own, i[k], 0 if V.block_shape else c)
        return index_sum(out, i)

    return arrayed((*V.value_shape, *(dimension,) * order), term)


def arrayed(shape, entry, idx=()):
    if len(idx) == len(shape):
        return entry(idx)
    return tensor(tuple(arrayed(shape, entry, (*idx, m)) for m in range(shape[len(idx)])))


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
    return clamp_table(tables[basix.index(*counts)].reshape(*X.shape[:-1], -1))


def clamp_table(values, *, rtol=1e-6, atol=1e-9):
    """Snap tabulated basis values close to -1, 0 or 1 to those exact values.

    The tolerances match FFCx. This numerical cleanup applies only to basis
    tabulation, including runtime tables, and leaves the input array unchanged.
    """
    values = np.asarray(values)
    for target in (-1.0, 0.0, 1.0):
        values = np.where(np.isclose(values, target, rtol=rtol, atol=atol), target, values)
    return values


def runtime_tables(e, env):
    # The library tabulates at the points that the solver gives, before the kernel runs.
    found = {}
    for n in runtime_arrays(e):
        match n:
            case RuntimeArray(Basis(V, j, points, d, c) as label, _):
                found[label] = basis(V.element(j), d, coordinates(points, env), c)
    return env | found
