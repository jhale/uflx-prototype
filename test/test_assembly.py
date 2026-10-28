import basix
import jax
import jax.numpy as jnp
import numpy as np
from test_basix import CELLS, mixed, square

from uflx_compiler import assemble, compile_form, evaluate, fold
from uflx_oracle import evaluate as oracle
from uflx_spec_interpreter import (
    form,
    function,
    gateaux_derivative,
    gradient,
    integral,
    normal,
    num,
    pull_back,
    side_map,
)
from uflx_toy.basix_cells import basix_cell
from uflx_toy.basix_spaces import basix_space

jax.config.update("jax_enable_x64", True)
P = basix.ElementFamily.P


def dot(a, b, D):
    g, h = gradient(a, D), gradient(b, D)
    return g[0] * h[0] + g[1] * h[1]


def test_the_mass_and_stiffness_matrices_of_two_triangles():
    D = square(basix_cell(CELLS[2]))[0]
    V = basix_space("V", D, P, 1)
    u, v = function(V, "u"), function(V, "v")
    M = assemble(form(integral(u * v, D), (v, u)), 2)({})
    K = assemble(form(integral(dot(u, v, D), D), (v, u)), 2)({})
    expected = np.array([[4, 1, 2, 1], [1, 2, 1, 0], [2, 1, 4, 1], [1, 0, 1, 2]]) / 24
    assert np.allclose(M, expected) and np.isclose(M.sum(), 1)
    stiffness = [[1, -0.5, 0, -0.5], [-0.5, 1, -0.5, 0], [0, -0.5, 1, -0.5], [-0.5, 0, -0.5, 1]]
    assert np.allclose(K, stiffness) and np.allclose(K @ np.ones(4), 0)


def interior_penalty(u, v, D, E):
    # The terms of the symmetric interior penalty method on the interior facets.
    plus, minus = side_map(D, E, "+"), side_map(D, E, "-")
    n = normal(plus)

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
    V = basix_space("V", D, P, 1, discontinuous=True)
    u, v, w, z = (function(V, n) for n in "uvwz")
    A = assemble(form(interior_penalty(u, v, D, E), (v, u)), 3)({})
    rng = np.random.default_rng(9)
    W, Z = rng.normal(size=V.num_dofs), rng.normal(size=V.num_dofs)
    expected = oracle(interior_penalty(z, w, D, E), {w: W, z: Z})(())
    assert np.isclose(W @ A @ Z, expected, rtol=1e-8)
    assert np.allclose(A, A.T)
    assert np.allclose(assemble(form(interior_penalty(u, v, D, E), (v, u)), 3)({}, jnp), A)


def test_the_gateaux_derivatives_assemble_to_the_derivatives_of_the_energy():
    # The residual and the tangent of the core are the gradient and the Hessian of the energy.
    D = mixed()[0]
    V = basix_space("V", D, P, 2)
    u, v, w = (function(V, n) for n in "uvw")
    energy = integral(num(0.5) * dot(u, u, D) + num(0.25) * u * u * u * u, D)
    residual = gateaux_derivative(energy, u, u, v)
    tangent = gateaux_derivative(residual, u, u, w)
    key = "coefficients:u"
    U = np.random.default_rng(10).normal(size=V.num_dofs) * 0.5
    compiled = compile_form(energy, 8)
    folded = fold(compiled)
    f = lambda U: evaluate(compiled, {key: U}, jnp, folded)
    assert np.allclose(assemble(residual, 8)({key: U}), jax.grad(f)(jnp.asarray(U)), atol=1e-10)
    assert np.allclose(assemble(tangent, 8)({key: U}), jax.hessian(f)(jnp.asarray(U)), atol=1e-10)
