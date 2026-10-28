import basix
import numpy as np
import pytest
from testutil import table

from uflx_oracle import at, evaluate
from uflx_spec_interpreter import (
    RD,
    cell_index,
    component_tensor,
    coordinate,
    decomposed_domain,
    domain_map,
    function,
    gradient,
    index,
    indexed,
    integral,
    normal,
    pull_back,
    side_domain,
    side_map,
)
from uflx_spec_interpreter._src.passes import gradients, integrals, pullbacks
from uflx_toy.basix_cells import basix_cell
from uflx_toy.cells import reference_simplex
from uflx_toy.inclusions import affine_map, reference_normal
from uflx_toy.spaces import space
from uflx_toy.topology import static_topology

CELLS = {1: basix.CellType.interval, 2: basix.CellType.triangle, 3: basix.CellType.tetrahedron}


def value(e, P=None, X=()):
    return np.real(at(P, np.asarray(X, dtype=float), lambda: evaluate(e, {})(())))


@pytest.mark.parametrize("n", [1, 2, 3])
def test_a_basix_simplex_is_the_unit_simplex(n):
    K = basix_cell(CELLS[n])
    assert (K.dimension, K.num_vertices) == (n, n + 1)
    assert np.allclose(value(K.vertices), value(reference_simplex(n).vertices))


@pytest.mark.parametrize("n", [2, 3])
def test_the_tabulated_facet_normals_are_the_cofactor_normals(n):
    K, F = basix_cell(CELLS[n]), reference_simplex(n - 1)
    V = value(K.vertices)
    for f, facet in enumerate(basix.cell.topology(CELLS[n])[n - 1]):
        points = table([[float(x) for x in V[v]] for v in facet])
        A = domain_map(F, affine_map(F, points), K)
        m = value(reference_normal(A, points, K), F, np.full(n - 1, 0.2))
        tabulated = value(K.facet_normals)[f]
        assert np.allclose(m / np.linalg.norm(m), tabulated)


def square(K):
    # The square of two triangles on the reference cell K, with its diagonal and boundary.
    I = basix_cell(basix.CellType.interval) if K == basix_cell(CELLS[2]) else reference_simplex(1)
    T = static_topology((K,), (((0, 1, 2), (0, 2, 3)),), ())
    c, k = cell_index(T, 0), index("k")
    p = table([[(0, 0), (1, 0), (1, 1)], [(0, 0), (1, 1), (0, 1)]])
    X = coordinate(K)
    vertex = lambda v: indexed(p, (c, v, k))
    edge = lambda v, d: (vertex(v) - vertex(0)) * indexed(X, (d,))
    D = decomposed_domain(
        (domain_map(K, component_tensor(vertex(0) + edge(1, 0) + edge(2, 1), (k,)), RD(2)),)
    )
    diagonal = static_topology((I,), (((0, 2),),), ((("+", T, 0), ("-", T, 0)),))
    boundary = static_topology((I,), (((0, 1), (1, 2), (2, 3), (3, 0)),), ((("inside", T, 0),),))
    return D, side_domain(D, diagonal), side_domain(D, boundary)


def facet_integrals(K):
    D, E, B = square(K)
    u = function(space("V", domain=D), "u")
    plus, minus, inside = side_map(D, E, "+"), side_map(D, E, "-"), side_map(D, B, "inside")
    env = {u: {0: lambda x: 1 + x[0] + 2 * x[1], 1: lambda x: 3 - x[0] * x[1]}}
    forms = [
        integral(pull_back(u, plus) * normal(plus)[0], E),
        integral(pull_back(u, plus) - pull_back(u, minus), E),
        integral(normal(inside)[1] * pull_back(coordinate(D)[0], inside), B),
        integral(pull_back(gradient(u, D)[0], inside) * normal(inside)[0], B),
    ]
    values = []
    for e in forms:
        out = gradients.apply_gradients(pullbacks.expand_pull_backs(integrals.expand_integrals(e)))
        assert gradients.in_n6(out)
        values.append(evaluate(out, env)(()))
        assert np.isclose(evaluate(e, env)(()), values[-1], rtol=1e-6)
    return np.array(values)


def test_a_mesh_of_basix_triangles_integrates_as_one_of_unit_simplices():
    assert np.allclose(facet_integrals(basix_cell(CELLS[2])), facet_integrals(reference_simplex(2)))


def test_a_basix_cell_gives_its_facet_normals_as_a_table():
    # A table read through the local facet has no support, a cofactor normal the support F.
    for K, constant in ((basix_cell(CELLS[2]), True), (reference_simplex(2), False)):
        _, E, _ = square(K)
        _, _, m = E.topology.side_tables(0, "+")
        assert (m.support is None) == constant and m.value_shape == (2,)
