import basix
import numpy as np
import pytest
from testutil import table, vertex_mesh

from uflx_oracle import at, evaluate
from uflx_spec_interpreter import (
    coordinate,
    domain_map,
    function,
    gradient,
    integral,
    normal,
    num,
    pull_back,
    side_domain,
    side_map,
)
from uflx_spec_interpreter._src.passes import gradients, integrals
from uflx_toy.basix_cells import basix_cell
from uflx_toy.cells import reference_simplex
from uflx_toy.inclusions import affine_map, reference_normal
from uflx_toy.spaces import space
from uflx_toy.topology import static_topology


def value(e, P=None, X=()):
    return np.real(at(P, np.asarray(X, dtype=float), lambda: evaluate(e, {})(())))


@pytest.mark.parametrize(
    ("n", "cell"),
    [
        (1, basix.CellType.interval),
        (2, basix.CellType.triangle),
        (3, basix.CellType.tetrahedron),
    ],
)
def test_a_basix_simplex_is_the_unit_simplex(n, cell):
    K = basix_cell(cell)
    assert (K.dimension, K.num_vertices) == (n, n + 1)
    assert np.allclose(value(K.vertices), value(reference_simplex(n).vertices))


@pytest.mark.parametrize(
    ("n", "cell"), [(2, basix.CellType.triangle), (3, basix.CellType.tetrahedron)]
)
def test_the_tabulated_facet_normals_are_the_cofactor_normals(n, cell):
    K, F = basix_cell(cell), reference_simplex(n - 1)
    V = value(K.vertices)
    for f, facet in enumerate(basix.cell.topology(cell)[n - 1]):
        points = table([[float(x) for x in V[v]] for v in facet])
        A = domain_map(F, affine_map(F, points), K)
        m = value(reference_normal(A, points, K), F, np.full(n - 1, 0.2))
        tabulated = value(K.facet_normals)[f]
        assert np.allclose(m / np.linalg.norm(m), tabulated)


def square(K):
    # The square of two triangles on the reference cell K, with its diagonal and boundary.
    I = (
        basix_cell(basix.CellType.interval)
        if K == basix_cell(basix.CellType.triangle)
        else reference_simplex(1)
    )
    T = static_topology((K,), (((0, 1, 2), (0, 2, 3)),), ())
    D = vertex_mesh(T, [(0, 0), (1, 0), (1, 1), (0, 1)])
    diagonal = static_topology((I,), (((0, 2),),), ((("+", T, 0), ("-", T, 0)),))
    boundary = static_topology((I,), (((0, 1), (1, 2), (2, 3), (3, 0)),), ((("inside", T, 0),),))
    return (
        D,
        side_domain(D, diagonal, diagonal.sides[0][0][0]),
        side_domain(D, boundary, boundary.sides[0][0][0]),
    )


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
        out = gradients._lower(integrals.expand_integrals(e))
        assert gradients._in_n6(out)
        values.append(evaluate(out, env)(()))
        assert np.isclose(evaluate(e, env)(()), values[-1], rtol=1e-6)
    return np.array(values)


def test_a_mesh_of_basix_triangles_integrates_as_one_of_unit_simplices():
    assert np.allclose(
        facet_integrals(basix_cell(basix.CellType.triangle)), facet_integrals(reference_simplex(2))
    )


def test_a_basix_cell_gives_its_facet_normals_as_a_table():
    # A table read through the local facet has no support, a cofactor normal the support F.
    for K, constant in ((basix_cell(basix.CellType.triangle), True), (reference_simplex(2), False)):
        _, E, _ = square(K)
        _, _, m = E.topology.side_tables(0, "+")
        assert (m.domain is None) == constant and m.value_shape == (2,)


def quadrilaterals():
    # The unit square as two quadrilaterals split at x = 1/2, with bilinear cell maps.
    Q, I = basix_cell(basix.CellType.quadrilateral), basix_cell(basix.CellType.interval)
    T = static_topology((Q,), (((0, 1, 3, 4), (1, 2, 4, 5)),), ())
    D = vertex_mesh(T, [(0, 0), (0.5, 0), (1, 0), (0, 1), (0.5, 1), (1, 1)])
    interior = static_topology((I,), (((1, 4),),), ((("+", T, 0), ("-", T, 0)),))
    boundary = static_topology(
        (I,), (((0, 1), (1, 2), (2, 5), (5, 4), (4, 3), (3, 0)),), ((("inside", T, 0),),)
    )
    return (
        D,
        side_domain(D, interior, interior.sides[0][0][0]),
        side_domain(D, boundary, boundary.sides[0][0][0]),
    )


def test_a_mesh_of_quadrilaterals_integrates_through_the_passes():
    D, E, B = quadrilaterals()
    u = function(space("V", domain=D), "u")
    plus, minus, inside = side_map(D, E, "+"), side_map(D, E, "-"), side_map(D, B, "inside")
    env = {u: {0: lambda x: 1 + x[0] + 2 * x[1], 1: lambda x: 3 - x[0] * x[1]}}
    Z = coordinate(D)
    exact = [
        (integral(num(1), D), 1),
        (integral(normal(plus)[0], E), 1),
        (integral(normal(minus)[0], E), -1),
        (integral(normal(inside)[0] * pull_back(Z[0], inside), B), 1),
        (integral(normal(inside)[1] * pull_back(Z[1], inside), B), 1),
        (integral(pull_back(u, plus) - pull_back(u, minus), E), -0.25),
        (integral(pull_back(gradient(u, D)[0], inside) * normal(inside)[0], B), None),
    ]
    for e, expected in exact:
        out = gradients._lower(integrals.expand_integrals(e))
        assert gradients._in_n6(out)
        before, after = evaluate(e, env)(()), evaluate(out, env)(())
        assert np.isclose(before, after, rtol=1e-6), (before, after)
        if expected is not None:
            assert np.isclose(after, expected, rtol=1e-6), (after, expected)


def mixed():
    # The unit square as two triangles left of x = 1/2 and one quadrilateral right of it.
    Tri, Q, I = (
        basix_cell(t)
        for t in (basix.CellType.triangle, basix.CellType.quadrilateral, basix.CellType.interval)
    )
    T = static_topology((Tri, Q), (((0, 1, 4), (0, 4, 3)), ((1, 2, 4, 5),)), ())
    D = vertex_mesh(T, [(0, 0), (0.5, 0), (1, 0), (0, 1), (0.5, 1), (1, 1)])
    # A group of facets between two triangles, and one between a triangle and a quadrilateral.
    interior = static_topology(
        (I, I),
        (((0, 4),), ((1, 4),)),
        ((("+", T, 0), ("-", T, 0)), (("+", T, 0), ("-", T, 1))),
    )
    boundary = static_topology(
        (I, I),
        (((0, 1), (4, 3), (3, 0)), ((1, 2), (2, 5), (5, 4))),
        ((("inside", T, 0),), (("inside", T, 1),)),
    )
    return (
        D,
        side_domain(D, interior, interior.sides[0][0][0]),
        side_domain(D, boundary, boundary.sides[0][0][0]),
    )


def test_a_mesh_of_triangles_and_a_quadrilateral():
    D, E, B = mixed()
    assert D.topology is not None and len(D.pieces) == 2 and len(E.pieces) == 2
    u = function(space("V", domain=D), "u")
    plus, minus, inside = side_map(D, E, "+"), side_map(D, E, "-"), side_map(D, B, "inside")
    # A function that differs between the cells, keyed by the cell type and the cell.
    env = {
        u: {
            (0, 0): lambda x: 1 + x[0] + 2 * x[1],
            (0, 1): lambda x: 2 - x[0],
            (1, 0): lambda x: 3 - x[0] * x[1],
        }
    }
    Z = coordinate(D)
    exact = [
        (integral(num(1), D), 1),
        (integral(normal(plus)[0], E), 0),
        (integral(normal(plus)[1], E), 0.5),
        (integral(normal(inside)[0] * pull_back(Z[0], inside), B), 1),
        (integral(normal(inside)[1] * pull_back(Z[1], inside), B), 1),
        (integral(u * u, D), None),
        (integral(pull_back(u, plus) - pull_back(u, minus), E), None),
        (integral(pull_back(gradient(u, D)[0], inside) * normal(inside)[0], B), None),
    ]
    for e, expected in exact:
        out = gradients._lower(integrals.expand_integrals(e))
        assert gradients._in_n6(out)
        before, after = evaluate(e, env)(()), evaluate(out, env)(())
        assert np.isclose(before, after, rtol=1e-6, atol=1e-9), (before, after)
        if expected is not None:
            assert np.isclose(after, expected, rtol=1e-6, atol=1e-9), (after, expected)
