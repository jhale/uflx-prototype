import basix
import jax
import jax.numpy as jnp
import numpy as np
from test_basix import mixed, square
from testutil import push_forward, side_map

from uflx_compiler import assemble, assemble_bound, compile_form, evaluate, fold
from uflx_oracle import evaluate as oracle
from uflx_spec_interpreter import (
    directional_derivative,
    form,
    function,
    gradient,
    integral,
    num,
    pull_back,
)
from uflx_toy.assembly import conforming_dofmap
from uflx_toy.basix_cells import basix_cell
from uflx_toy.basix_spaces import basix_space
from uflx_toy.sides import outward_conormal

jax.config.update("jax_enable_x64", True)
P = basix.ElementFamily.P


def dot(a, b, D):
    g, h = gradient(a, D), gradient(b, D)
    return g[0] * h[0] + g[1] * h[1]


def test_the_mass_and_stiffness_matrices_of_two_triangles():
    D = square(basix_cell(basix.CellType.triangle))[0]
    V = basix_space("V", D.topology, P, 1)
    uT, vT = function(V, "u"), function(V, "v")
    u, v = push_forward(uT, D), push_forward(vT, D)
    mapping = conforming_dofmap(V)
    bindings = {uT: mapping, vT: mapping}
    M = assemble_bound(form(integral(u * v, D), (vT, uT)), 2, bindings)({})
    K = assemble_bound(form(integral(dot(u, v, D), D), (vT, uT)), 2, bindings)({})
    expected = np.array([[4, 1, 2, 1], [1, 2, 1, 0], [2, 1, 4, 1], [1, 0, 1, 2]]) / 24
    assert np.allclose(M, expected) and np.isclose(M.sum(), 1)
    stiffness = [[1, -0.5, 0, -0.5], [-0.5, 1, -0.5, 0], [0, -0.5, 1, -0.5], [-0.5, 0, -0.5, 1]]
    assert np.allclose(K, stiffness) and np.allclose(K @ np.ones(4), 0)


def interior_penalty(u, v, D, E):
    # The terms of the symmetric interior penalty method on the interior facets.
    plus, minus = side_map(D, E, "+"), side_map(D, E, "-")
    n = outward_conormal(plus)

    def jump(f):
        return pull_back(f, plus) - pull_back(f, minus)

    def average_normal(f):
        g = gradient(f, D)
        return (
            pull_back(g[0], plus) * n[0]
            + pull_back(g[1], plus) * n[1]
            + pull_back(g[0], minus) * n[0]
            + pull_back(g[1], minus) * n[1]
        ) * num(0.5)

    facets = -average_normal(u) * jump(v) - jump(u) * average_normal(v) + num(4) * jump(u) * jump(v)
    return integral(dot(u, v, D), D) + integral(facets, E)


def test_an_assembled_bilinear_form_with_facet_terms_on_a_mixed_mesh():
    D, E, _ = mixed()
    V = basix_space("V", D.topology, P, 1, discontinuous=True)
    uT, vT, wT, zT = (function(V, n) for n in "uvwz")
    u, v, w, z = (push_forward(f, D) for f in (uT, vT, wT, zT))
    A = assemble(form(interior_penalty(u, v, D, E), (vT, uT)), 3)({})
    rng = np.random.default_rng(9)
    W, Z = rng.normal(size=V.num_dofs), rng.normal(size=V.num_dofs)
    expected = oracle(interior_penalty(z, w, D, E), {wT: W, zT: Z})(())
    assert np.isclose(W @ A @ Z, expected, rtol=1e-8)
    assert np.allclose(A, A.T)
    assert np.allclose(assemble(form(interior_penalty(u, v, D, E), (vT, uT)), 3)({}, jnp), A)


def test_bound_directional_derivatives_assemble_to_the_energy_derivatives():
    # The residual and the tangent of the core are the gradient and the Hessian of the energy.
    D = mixed()[0]
    V = basix_space("V", D.topology, P, 2)
    uT, vT, wT = (function(V, n) for n in "uvw")
    u = push_forward(uT, D)
    energy = integral(num(0.5) * dot(u, u, D) + num(0.25) * u * u * u * u, D)
    residual = form(directional_derivative(energy, uT, uT, vT), (vT,))
    tangent = form(directional_derivative(residual, uT, uT, wT), (wT,))
    key = "coefficients:u"
    U = np.random.default_rng(10).normal(size=V.num_dofs) * 0.5
    compiled = compile_form(energy, 8)
    folded = fold(compiled)
    f = lambda U: evaluate(compiled, {key: U}, jnp, folded)
    assert np.allclose(assemble(residual, 8)({key: U}), jax.grad(f)(jnp.asarray(U)), atol=1e-10)
    assert np.allclose(assemble(tangent, 8)({key: U}), jax.hessian(f)(jnp.asarray(U)), atol=1e-10)


def test_zero_form_uses_argument_spaces_for_rectangular_complex_tensor():
    from uflx_spec_interpreter import ScalarType

    D = square(basix_cell(basix.CellType.triangle))[0]
    V = basix_space("rows", D.topology, P, 1, True, ScalarType.complex)
    W = basix_space("columns", D.topology, P, 2, True, ScalarType.complex)
    v, u = function(V, "v"), function(W, "u")
    f = form(integral(num(0j), D), (v, u))
    value = assemble(f, 2)({})
    assert value.shape == (V.num_dofs, W.num_dofs) == (6, 12)
    assert np.iscomplexobj(value)
    assert np.all(value == 0)
    np.testing.assert_array_equal(assemble(f, 2)({}, jnp), value)


def test_unused_argument_keeps_its_axis():
    D = square(basix_cell(basix.CellType.triangle))[0]
    V = basix_space("unused", D.topology, P, 1, True)
    v = function(V, "v")
    f = form(integral(num(0), D), (v,))
    value = assemble(f, 2)({})
    assert value.shape == (V.num_dofs,)
    assert np.all(value == 0)
