from math import factorial

import basix
import numpy as np
import pytest

from uflx._src.passes import integrals
from uflx._src.passes.contracts import in_core_normal_form
from uflx._src.passes.lowering import lower_spatial
from uflx.arithmetic import num
from uflx.calculus import coordinate, gradient, integral
from uflx.forms import function
from uflx.geometry import domain_map, pull_back
from uflx.passes import expand_arithmetic
from uflx_basix.cells import cell
from uflx_basix.interpolation import interpolate
from uflx_basix.spaces import space
from uflx_jax_compiler.interpreter import Interpreter

from ..testutil import table
from ..uflx.inclusions import affine_map, reference_normal
from ..uflx.sides import outward_conormal
from ..uflx.spaces import space as field_space
from ..uflx_basix.cells import reference_simplex
from ..uflx_basix.meshes import (
    mixed,
    quadrilaterals,
    side_data,
    side_map,
    square,
)


def value(e, P=None, X=()):
    points = {P: X} if P is not None else {}
    return np.real(Interpreter().evaluate(e, points=points))


def test_reference_interpolation_uses_the_element_sampling_points():
    K = cell(basix.CellType.triangle)
    V = space(K, basix.create_element(basix.ElementFamily.P, K.cell_type, 1))
    x = coordinate(K)[0]
    interpolant = interpolate(x * x, V)
    assert value(interpolant, K, (0.25, 0.5)) == pytest.approx(0.25)


@pytest.mark.parametrize(
    ("n", "cell_type"),
    [
        (1, basix.CellType.interval),
        (2, basix.CellType.triangle),
        (3, basix.CellType.tetrahedron),
    ],
)
def test_a_basix_simplex_is_the_unit_simplex(n, cell_type):
    K = cell(cell_type)
    assert (K.dimension, K.num_vertices) == (n, n + 1)
    simplex = reference_simplex(n)
    assert np.allclose(value(K.vertices), value(simplex.vertices))
    for domain in (simplex, K):
        moment = integral(coordinate(domain)[0] ** num(12), domain)
        assert value(moment) == pytest.approx(factorial(12) / factorial(12 + n), rel=1e-12)


@pytest.mark.parametrize(
    ("n", "cell_type"), [(2, basix.CellType.triangle), (3, basix.CellType.tetrahedron)]
)
def test_the_tabulated_facet_normals_are_the_cofactor_normals(n, cell_type):
    K, F = cell(cell_type), reference_simplex(n - 1)
    V = value(K.vertices)
    for f, facet in enumerate(basix.cell.topology(cell_type)[n - 1]):
        points = table([[float(x) for x in V[v]] for v in facet])
        A = domain_map(F, affine_map(F, points), K)
        m = value(reference_normal(A, points, K), F, np.full(n - 1, 0.2))
        tabulated = basix.cell.facet_outward_normals(cell_type)[f]
        assert np.allclose(m / np.linalg.norm(m), tabulated)


def facet_integrals(K):
    D, E, B, geometry = square(K)
    u = function(field_space("V", domain=D), "u")
    plus, minus, inside = side_map(D, E, "+"), side_map(D, E, "-"), side_map(D, B, "inside")
    x = coordinate(D)
    env = geometry | {u: {0: num(1) + x[0] + num(2) * x[1], 1: num(3) - x[0] * x[1]}}
    forms = [
        integral(pull_back(u, plus) * outward_conormal(plus)[0], E),
        integral(pull_back(u, plus) - pull_back(u, minus), E),
        integral(outward_conormal(inside)[1] * pull_back(coordinate(D)[0], inside), B),
        integral(pull_back(gradient(u, D)[0], inside) * outward_conormal(inside)[0], B),
    ]
    values = []
    for e in forms:
        out = lower_spatial(integrals.expand_integrals(e))
        out = expand_arithmetic(out)
        assert in_core_normal_form(out)
        oracle = Interpreter(env)
        values.append(oracle.evaluate(out))
        assert np.isclose(oracle.evaluate(e), values[-1], rtol=1e-6)
    return np.array(values)


def test_a_mesh_of_basix_triangles_integrates_as_one_of_unit_simplices():
    assert np.allclose(
        facet_integrals(cell(basix.CellType.triangle)), facet_integrals(reference_simplex(2))
    )


def test_library_conormals_are_separate_from_the_inclusion():
    for K in (cell(basix.CellType.triangle), reference_simplex(2)):
        _, E, _, _ = square(K)
        A, m = side_data(E.topology, 0, "+")
        vector = value(m, A.parameter_domain, (0.5,))
        assert m.value_shape == (2,) and A.ambient_domain == K
        assert np.all(np.isfinite(vector)) and np.linalg.norm(vector) > 0


def test_a_mesh_of_quadrilaterals_integrates_through_the_passes():
    D, E, B, geometry = quadrilaterals()
    u = function(field_space("V", domain=D), "u")
    plus, minus, inside = side_map(D, E, "+"), side_map(D, E, "-"), side_map(D, B, "inside")
    x = coordinate(D)
    env = geometry | {u: {0: num(1) + x[0] + num(2) * x[1], 1: num(3) - x[0] * x[1]}}
    exact = [
        (integral(num(1), D), 1),
        (integral(outward_conormal(plus)[0], E), 1),
        (integral(outward_conormal(minus)[0], E), -1),
        (integral(outward_conormal(inside)[0] * pull_back(x[0], inside), B), 1),
        (integral(outward_conormal(inside)[1] * pull_back(x[1], inside), B), 1),
        (integral(pull_back(u, plus) - pull_back(u, minus), E), -0.25),
        (integral(pull_back(gradient(u, D)[0], inside) * outward_conormal(inside)[0], B), None),
    ]
    for e, expected in exact:
        out = lower_spatial(integrals.expand_integrals(e))
        out = expand_arithmetic(out)
        assert in_core_normal_form(out)
        oracle = Interpreter(env)
        before, after = oracle.evaluate(e), oracle.evaluate(out)
        assert np.isclose(before, after, rtol=1e-6), (before, after)
        if expected is not None:
            assert np.isclose(after, expected, rtol=1e-6), (after, expected)


def test_a_mesh_of_triangles_and_a_quadrilateral():
    D, E, B, geometry = mixed()
    assert (
        D.topology is not None
        and len(D.topology.cell_types) == 3
        and len(E.topology.cell_types) == 2
    )
    u = function(field_space("V", domain=D), "u")
    plus, minus, inside = side_map(D, E, "+"), side_map(D, E, "-"), side_map(D, B, "inside")
    # A function that differs between the cells, keyed by the local cell slot.
    x = coordinate(D)
    env = geometry | {
        u: {
            0: num(1) + x[0] + num(2) * x[1],
            1: num(2) - x[0],
            2: num(3) - x[0] * x[1],
        }
    }
    exact = [
        (integral(num(1), D), 1),
        (integral(outward_conormal(plus)[0], E), 0),
        (integral(outward_conormal(plus)[1], E), 0.5),
        (integral(outward_conormal(inside)[0] * pull_back(x[0], inside), B), 1),
        (integral(outward_conormal(inside)[1] * pull_back(x[1], inside), B), 1),
        (integral(u * u, D), None),
        (integral(pull_back(u, plus) - pull_back(u, minus), E), None),
        (integral(pull_back(gradient(u, D)[0], inside) * outward_conormal(inside)[0], B), None),
    ]
    for e, expected in exact:
        out = lower_spatial(integrals.expand_integrals(e))
        out = expand_arithmetic(out)
        assert in_core_normal_form(out)
        oracle = Interpreter(env)
        before, after = oracle.evaluate(e), oracle.evaluate(out)
        assert np.isclose(before, after, rtol=1e-6, atol=1e-9), (before, after)
        if expected is not None:
            assert np.isclose(after, expected, rtol=1e-6, atol=1e-9), (after, expected)


def test_custom_element_supplies_its_own_interpolation_functional():
    from ..uflx_basix.test_spaces import interval_element

    K = cell(basix.CellType.interval)
    left = space(K, interval_element(0.25))
    right = space(K, interval_element(0.75))
    assert left is not right
    x = coordinate(left.domain)[0]
    assert value(interpolate(x, left), left.domain, (0.5,)) == pytest.approx(0.25)
    assert value(interpolate(x, right), right.domain, (0.5,)) == pytest.approx(0.75)


def test_oracle_tabulates_each_custom_elements_own_basis():
    from ..uflx_basix.test_spaces import interval_element

    point = np.array([[0.22]])
    for first_dof in (0.22, 0.22 + 1e-12):
        element = interval_element(first_dof, 0.7)
        V = space(cell(element.cell_type), element)
        u = function(V, "u")
        weights = np.array([0.0, 1e10])
        expected = element.tabulate(0, point)[0, 0, :, 0] @ weights
        actual = Interpreter({u: weights}).evaluate(u, points={V.domain: point[0]})
        assert actual == pytest.approx(expected, abs=1e-9)
