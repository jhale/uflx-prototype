import re

import basix
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from test_basix import CELLS, mixed, square
from testutil import contains

from uflx_compiler import compile_form, evaluate, fold
from uflx_oracle import evaluate as oracle
from uflx_spec_interpreter import function, gradient, integral, num
from uflx_spec_interpreter._src.nodes import Coordinate, Gradient, Integral, PullBack
from uflx_toy.basix_cells import basix_cell
from uflx_toy.basix_spaces import basix_space

jax.config.update("jax_enable_x64", True)
P = basix.ElementFamily.P


def functionals(D, u):
    g = gradient(u, D)
    return [
        integral(num(1), D),
        integral(u, D),
        integral(u * u, D),
        integral(g[0] * g[0] + g[1] * g[1], D),
        integral(u * g[0] * g[1], D),
    ]


@pytest.mark.parametrize(
    ("mesh", "degree"),
    [("triangles", 1), ("triangles", 2), ("mixed", 1), ("mixed", 2)],
)
def test_compiled_functionals_agree_with_the_oracle(mesh, degree):
    D = square(basix_cell(CELLS[2]))[0] if mesh == "triangles" else mixed()[0]
    V = basix_space("V", D, P, degree)
    u = function(V, "u")
    U = np.random.default_rng(degree).normal(size=V.num_dofs)
    for e in functionals(D, u):
        compiled = compile_form(e, 3 * degree + 2)
        # Quadrature and tabulation leave arithmetic on arrays, with no lazy node.
        lazy = (Integral, PullBack, Gradient, Coordinate)
        assert not contains(compiled, lambda n, lazy=lazy: isinstance(n, lazy))
        env = {("coefficients", u): U}
        expected = oracle(e, {u: U})(())
        assert np.isclose(evaluate(compiled, env), expected, rtol=1e-8), e
        assert np.isclose(evaluate(compiled, env, np, fold(compiled)), expected, rtol=1e-8), e


def test_the_same_graph_runs_on_jax():
    D = mixed()[0]
    V = basix_space("V", D, P, 2)
    u = function(V, "u")
    U = np.random.default_rng(5).normal(size=V.num_dofs)
    compiled = compile_form(functionals(D, u)[3], 8)
    key = ("coefficients", u)
    f = jax.jit(lambda U: evaluate(compiled, {key: U}, jnp))
    assert np.isclose(f(jnp.asarray(U)), evaluate(compiled, {key: U}))
    # The derivative in the coefficients is the residual vector, here by automatic differentiation.
    d = np.random.default_rng(6).normal(size=V.num_dofs)
    step = 1e-6
    central = evaluate(compiled, {key: U + step * d}) - evaluate(compiled, {key: U - step * d})
    assert np.isclose(jax.grad(f)(jnp.asarray(U)) @ d, central / (2 * step), rtol=1e-6)


def instructions(compiled, key, n, folded=None):
    f = jax.jit(lambda U: evaluate(compiled, {key: U}, jnp, folded))
    return len(re.findall(r"= \S+ \w[\w-]*\(", f.lower(jnp.zeros(n)).compile().as_text()))


def test_folding_the_geometry_shrinks_the_kernel():
    # The geometry of a mesh with constant coordinates reads no kernel input, so it is folded.
    D = mixed()[0]
    V = basix_space("V", D, P, 2)
    u = function(V, "u")
    compiled = compile_form(functionals(D, u)[3], 4)
    key = ("coefficients", u)
    folded = fold(compiled)
    assert compiled not in folded and folded
    plain, small = (
        instructions(compiled, key, V.num_dofs),
        instructions(compiled, key, V.num_dofs, folded),
    )
    assert small < plain / 3, (plain, small)
    U = np.random.default_rng(7).normal(size=V.num_dofs)
    f = jax.jit(lambda U: evaluate(compiled, {key: U}, jnp, folded))
    assert np.isclose(f(jnp.asarray(U)), evaluate(compiled, {key: U}))
