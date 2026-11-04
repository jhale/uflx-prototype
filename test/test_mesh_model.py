"""Concrete checks of mesh support versus explicit integration geometry."""

import basix
import numpy as np
from testutil import push_forward, table, vertex_mesh

from uflx_compiler import compile_form
from uflx_compiler import evaluate as compiled_value
from uflx_oracle import evaluate
from uflx_spec_interpreter import (
    RD,
    ScalarType,
    coordinate,
    density,
    domain_map,
    function,
    gradient,
    image,
    index,
    index_sum,
    integral,
    jacobian,
    num,
    pull_back,
    tensor,
)
from uflx_spec_interpreter._src.algorithms.substitution import _substitute
from uflx_spec_interpreter._src.meshes import _cell_domain
from uflx_spec_interpreter._src.nodes import Mesh
from uflx_spec_interpreter._src.passes.lowering import _lower_spatial
from uflx_toy.basix_cells import basix_cell
from uflx_toy.basix_spaces import basix_space
from uflx_toy.cells import reference_simplex
from uflx_toy.spaces import space
from uflx_toy.topology import static_topology


def test_integrate_members_then_sum_only_the_selected_index():
    P = reference_simplex(1)
    i, configuration = index("segment"), index("configuration")
    lengths = table([[2, 4], [3, 6], [5, 10]])
    M = domain_map(P, tensor((lengths[i, configuration] * coordinate(P)[0],)), RD(1))
    members = integral(pull_back(num(1), M) * density(M), P)
    total = index_sum(members, (i,))
    assert dict(members.free_indices) == {i: 3, configuration: 2}
    assert dict(total.free_indices) == {configuration: 2}
    for c, expected in enumerate((10, 20)):
        assert np.isclose(evaluate(_lower_spatial(total), {}, {configuration: c})(()), expected)


def test_integration_patches_need_no_field_domain():
    P = reference_simplex(1)
    t = coordinate(P)[0]
    patches = (
        domain_map(P, tensor((num(2) * t,)), RD(1)),
        domain_map(P, tensor((num(2) + num(3) * t,)), RD(1)),
    )
    f = coordinate(RD(1))[0]
    expression = sum((integral(pull_back(f, M) * density(M), P) for M in patches), num(0))
    assert np.isclose(evaluate(_lower_spatial(expression), {})(()), 12.5)


def test_coincident_cells_keep_independent_finite_element_fields():
    K = basix_cell(basix.CellType.interval)
    T = static_topology((K,), (((0, 1), (2, 3)),), ())
    D = vertex_mesh(T, [(0,), (1,), (0,), (1,)])
    V = basix_space("independent", T, basix.ElementFamily.P, 1)
    u = push_forward(function(V, "u"), D)
    coefficients = np.empty(V.num_dofs)
    for row, value in zip(V.dofmap[0], (2, 7), strict=True):
        coefficients[list(row)] = value
    result = compiled_value(compile_form(integral(u, D), 2), {"coefficients:u": coefficients})
    assert np.isclose(result, 9)
    assert D.topology is T


def test_mesh_identity_and_geometry_follow_coordinate_replacement():
    K = reference_simplex(1)
    T = static_topology((K,), (((0, 1),),), ())
    V = space("coordinate", value_shape=(1,), domain=T, scalar_type=ScalarType.real)
    x, y = function(V, "x"), function(V, "y")
    D, E = image(x, RD(1)), image(y, RD(1))
    assert isinstance(D, Mesh)
    assert D.construction_arguments == (x, RD(1))
    assert image(x, RD(1)) is D
    assert _substitute(D, {x: y}) is E
    assert _substitute(jacobian(_cell_domain(D, 0)), {x: y}) is jacobian(_cell_domain(E, 0))


def test_diffusion_on_independent_analytic_curve_patches():
    # Two patch charts x_a(r) = (r, a*r**2), with a=1 and a=2.
    # No shared vertices or sides, and no new coordinate-expression class.
    K = basix_cell(basix.CellType.interval)
    T = static_topology((K,), (((0, 1), (2, 3)),), ())
    R = basix_space("patch-coordinate", T, basix.ElementFamily.P, 1)
    A = basix_space("patch-curvature", T, basix.ElementFamily.P, 0, discontinuous=True)
    r, a = function(R, "r"), function(A, "a")
    D = image(tensor((r, a * r * r)), RD(2))
    u = push_forward(r, D)
    g = gradient(u, D)
    energy = integral(g[0] * g[0] + g[1] * g[1], D)
    r_values = np.empty(R.num_dofs)
    a_values = np.empty(A.num_dofs)
    for row in R.dofmap[0]:
        r_values[list(row)] = (0, 1)
    for row, curvature in zip(A.dofmap[0], (1, 2), strict=True):
        a_values[list(row)] = curvature
    env = {"coefficients:r": r_values, "coefficients:a": a_values}
    value = compiled_value(compile_form(energy, 40), env)
    # |grad_surface r|^2 ds = dr/sqrt(1 + 4*a*a*r*r).
    expected = sum(np.arcsinh(2 * curvature) / (2 * curvature) for curvature in (1, 2))
    assert np.isclose(value, expected, rtol=1e-10, atol=1e-12)


def test_mesh_rejects_free_coordinate_indices_but_keeps_symbolic_coordinates():
    import pytest

    K = reference_simplex(1)
    T = static_topology((K,), (((0, 1),),), ())
    D = vertex_mesh(T, [(0,), (1,)])
    q = index("configuration")
    coordinates = table([2, 3])[q] * D.coordinate_function
    with pytest.raises(ValueError, match="no free indices"):
        image(coordinates, RD(1))
    with pytest.raises(ValueError, match="no free indices"):
        Mesh(coordinates, RD(1))
    # A symbolic scalar parameter is fine; only an unbound family index is excluded.
    scale = function(space("scale", scalar_type=ScalarType.real), "scale")
    symbolic = image(scale * D.coordinate_function, RD(1))
    assert _substitute(symbolic, {scale: num(2)}) is image(num(2) * D.coordinate_function, RD(1))
    with pytest.raises(ValueError, match="no free indices"):
        _substitute(symbolic, {scale: table([2, 3])[q]})


def test_mesh_integral_retains_integrand_indices_across_substitution():
    from uflx_spec_interpreter import cell_index
    from uflx_spec_interpreter._src.passes.integrals import expand_integrals

    K = reference_simplex(1)
    T = static_topology((K,), (((0, 1), (2, 3)),), ())
    D = vertex_mesh(T, [(0,), (1,), (0,), (1,)])
    c = cell_index(T, 0)
    r = function(space("load"), "r")
    load = table([2, 7])[c]
    expression = integral(r, D)
    inserted = _substitute(expression, {r: load})
    first = expand_integrals(inserted)
    second = _substitute(expand_integrals(expression), {r: load})
    assert inserted.free_indices == first.free_indices == second.free_indices == ((c, 2),)
    for n, expected in enumerate((4, 14)):
        for form in (inserted, first, second):
            assert np.isclose(evaluate(_lower_spatial(form), {}, {c: n})(()), expected)
    assert expand_integrals(inserted) is first
