"""Concrete checks of mesh support versus explicit integration geometry."""

import basix
import numpy as np

from uflx._src.algorithms.substitution import substitute
from uflx._src.meshes import cell_map
from uflx._src.nodes import Mesh
from uflx._src.passes.lowering import lower_spatial
from uflx.arithmetic import num
from uflx.calculus import coordinate, gradient, integral
from uflx.forms import function
from uflx.geometry import density, domain_map, jacobian, mesh, pull_back
from uflx.nodes import EuclideanSpace, ScalarType
from uflx.tensors import index, index_sum, tensor
from uflx_basix.basix_cells import basix_cell
from uflx_basix.basix_spaces import basix_space
from uflx_basix.cells import reference_simplex
from uflx_basix.spaces import space
from uflx_compiler import compile_form
from uflx_compiler import evaluate as compiled_value
from uflx_oracle import evaluate

from ..testutil import grouped_topology, push_forward, table, vertex_mesh


def test_integrate_members_then_sum_only_the_selected_index():
    P = reference_simplex(1)
    i, configuration = index("segment"), index("configuration")
    lengths = table([[2, 4], [3, 6], [5, 10]])
    M = domain_map(P, tensor((lengths[i, configuration] * coordinate(P)[0],)), EuclideanSpace(1))
    members = integral(pull_back(num(1), M) * density(M), P)
    total = index_sum(members, (i,))
    assert dict(members.free_indices) == {i: 3, configuration: 2}
    assert dict(total.free_indices) == {configuration: 2}
    for c, expected in enumerate((10, 20)):
        assert np.isclose(evaluate(lower_spatial(total), {}, {configuration: c})(()), expected)


def test_integration_patches_need_no_field_domain():
    P = reference_simplex(1)
    t = coordinate(P)[0]
    patches = (
        domain_map(P, tensor((num(2) * t,)), EuclideanSpace(1)),
        domain_map(P, tensor((num(2) + num(3) * t,)), EuclideanSpace(1)),
    )
    f = coordinate(EuclideanSpace(1))[0]
    expression = sum((integral(pull_back(f, M) * density(M), P) for M in patches), num(0))
    assert np.isclose(evaluate(lower_spatial(expression), {})(()), 12.5)


def test_coincident_cells_keep_independent_finite_element_fields():
    K = basix_cell(basix.CellType.interval)
    T = grouped_topology((K,), (((0, 1), (2, 3)),), ())
    D = vertex_mesh(T, [(0,), (1,), (0,), (1,)])
    V = basix_space("independent", T, basix.ElementFamily.P, 1)
    u = push_forward(function(V, "u"), D)
    coefficients = np.empty(V.num_dofs)
    for row, value in zip(V.cell_dofs, (2, 7), strict=True):
        coefficients[list(row)] = value
    result = compiled_value(compile_form(integral(u, D), 2), {"coefficients:u": coefficients})
    assert np.isclose(result, 9)
    assert D.topology is T


def test_mesh_identity_and_geometry_follow_coordinate_replacement():
    K = reference_simplex(1)
    T = grouped_topology((K,), (((0, 1),),), ())
    V = space("coordinate", value_shape=(1,), domain=T, scalar_type=ScalarType.real)
    x, y = function(V, "x"), function(V, "y")
    D, E = mesh(x, EuclideanSpace(1)), mesh(y, EuclideanSpace(1))
    assert isinstance(D, Mesh)
    assert D.construction_arguments == (x, EuclideanSpace(1))
    assert mesh(x, EuclideanSpace(1)) is D
    assert substitute(D, {x: y}) is E
    assert substitute(jacobian(cell_map(D, 0)), {x: y}) is jacobian(cell_map(E, 0))


def test_diffusion_on_independent_analytic_curve_patches():
    # Two patch charts x_a(r) = (r, a*r**2), with a=1 and a=2.
    # No shared vertices or sides, and no new coordinate-expression class.
    K = basix_cell(basix.CellType.interval)
    T = grouped_topology((K,), (((0, 1), (2, 3)),), ())
    R = basix_space("patch-coordinate", T, basix.ElementFamily.P, 1)
    A = basix_space("patch-curvature", T, basix.ElementFamily.P, 0, discontinuous=True)
    r, a = function(R, "r"), function(A, "a")
    D = mesh(tensor((r, a * r * r)), EuclideanSpace(2))
    u = push_forward(r, D)
    g = gradient(u, D)
    energy = integral(g[0] * g[0] + g[1] * g[1], D)
    r_values = np.empty(R.num_dofs)
    a_values = np.empty(A.num_dofs)
    for row in R.cell_dofs:
        r_values[list(row)] = (0, 1)
    for row, curvature in zip(A.cell_dofs, (1, 2), strict=True):
        a_values[list(row)] = curvature
    env = {"coefficients:r": r_values, "coefficients:a": a_values}
    value = compiled_value(compile_form(energy, 40), env)
    # |grad_surface r|^2 ds = dr/sqrt(1 + 4*a*a*r*r).
    expected = sum(np.arcsinh(2 * curvature) / (2 * curvature) for curvature in (1, 2))
    assert np.isclose(value, expected, rtol=1e-10, atol=1e-12)


def test_mesh_rejects_free_coordinate_indices_but_keeps_symbolic_coordinates():
    import pytest

    K = reference_simplex(1)
    T = grouped_topology((K,), (((0, 1),),), ())
    D = vertex_mesh(T, [(0,), (1,)])
    q = index("configuration")
    coordinates = table([2, 3])[q] * D.coordinate_function
    with pytest.raises(ValueError, match="no free indices"):
        mesh(coordinates, EuclideanSpace(1))
    with pytest.raises(ValueError, match="no free indices"):
        Mesh(coordinates, EuclideanSpace(1))
    # A symbolic scalar parameter is fine; only an unbound family index is excluded.
    scale = function(space("scale", scalar_type=ScalarType.real), "scale")
    symbolic = mesh(scale * D.coordinate_function, EuclideanSpace(1))
    assert substitute(symbolic, {scale: num(2)}) is mesh(
        num(2) * D.coordinate_function, EuclideanSpace(1)
    )
    with pytest.raises(ValueError, match="no free indices"):
        substitute(symbolic, {scale: table([2, 3])[q]})


def test_mesh_integral_retains_integrand_indices_across_substitution():
    from uflx._src.passes.integrals import expand_integrals

    K = reference_simplex(1)
    T = grouped_topology((K,), (((0, 1), (2, 3)),), ())
    D = vertex_mesh(T, [(0,), (1,), (0,), (1,)])
    c = index("load-entry")
    r = function(space("load"), "r")
    load = table([2, 7])[c]
    expression = integral(r, D)
    inserted = substitute(expression, {r: load})
    first = expand_integrals(inserted)
    second = substitute(expand_integrals(expression), {r: load})
    assert inserted.free_indices == first.free_indices == second.free_indices == ((c, 2),)
    for n, expected in enumerate((4, 14)):
        for form in (inserted, first, second):
            assert np.isclose(evaluate(lower_spatial(form), {}, {c: n})(()), expected)
    assert expand_integrals(inserted) is first
