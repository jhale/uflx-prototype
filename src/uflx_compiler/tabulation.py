import numpy as np

from uflx_spec_interpreter import (
    RD,
    index,
    index_sum,
    tensor,
)
from uflx_spec_interpreter._src.domains import cell_of
from uflx_spec_interpreter._src.nodes import (
    ComponentTensor,
    DomainMap,
    Function,
    GatheredIndex,
    Gradient,
    Indexed,
    PullBack,
    Tensor,
)
from uflx_spec_interpreter._src.substitution import bottom_up
from uflx_toy.basix_nodes import BasixCell, BasixSpace
from uflx_toy.nodes import IndexTable, RuntimeArray
from uflx_toy.runtime import runtime_array

from .evaluation import evaluate
from .inputs import Basis, runtime_arrays
from .quadrature import constant


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
    # The gathered coefficients of the cell times a table for each factor of the element, summed
    # over a dof index for each factor. An element that does not factor is its one factor.
    V, (j, entry) = u.space, found_cell
    dimension = points.value_shape[0]
    if V.factors(j) and separable(points):
        factors = [(F, X, (k,)) for k, (F, X) in enumerate(zip(V.factors(j), points.entries))]
    else:
        factors = [(V.element(j), points, tuple(range(dimension)))]
    i = tuple(index(("compiler.tabulation", u, entry, points, k)) for k in range(len(factors)))
    rows = np.asarray(V.dofmap[j]).reshape(-1, *(F.dim for F, _, _ in factors))
    coefficient = coefficients(u)[GatheredIndex(IndexTable(nested(rows), V.num_dofs), (entry, *i))]

    def term(direction):
        out = coefficient
        for k, (F, X, covered) in enumerate(factors):
            d = covered.index(direction) if direction in covered else None
            axes = [n for n, _ in X.free_indices]
            out = out * basis_table(V, j, F, X, d)[(*axes, i[k])]
        return index_sum(out, i)

    if derivative == 0:
        return term(None)
    return tensor(tuple(term(m) for m in range(dimension)))


def separable(points):
    # Coordinate k of the points reads indices that no other coordinate reads.
    match points:
        case Tensor(entries) if not runtime_arrays(points):
            seen = [set(dict(x.free_indices)) for x in entries]
            return sum(map(len, seen)) == len(set().union(*seen))
    return False


def nested(values):
    return tuple(map(nested, values)) if isinstance(values, np.ndarray) else int(values)


def basis_table(V, j, F, X, d):
    # An axis for each free index of the points X, such as the point and the facet, then the dof.
    # Points known only at assembly give a table that runtime_tables fills.
    if not runtime_arrays(X):
        return constant(basis(F, d, coordinates(X, {})))
    extents = [m for _, m in X.free_indices]
    return runtime_array(Basis(V, j, X, d), (*extents, F.dim))


def coordinates(X, env):
    # The points as an array with the coordinate last, which a factor reads one at a time.
    values = np.real(np.asarray(evaluate(X, env, np)))
    return values if X.value_shape else values[..., None]


def basis(F, d, X):
    # The basis, or its reference derivative in direction d, at points of any leading shape.
    tables = F.tabulate(0 if d is None else 1, X.reshape(-1, X.shape[-1]))[..., 0]
    return tables[0 if d is None else 1 + d].reshape(*X.shape[:-1], -1)


def runtime_tables(e, env):
    # The library tabulates at the points that the solver gives, before the kernel runs.
    found = {}
    for n in runtime_arrays(e):
        match n:
            case RuntimeArray(Basis(V, j, points, d) as label, _):
                found[label] = basis(V.element(j), d, coordinates(points, env))
    return env | found
