import re

import basix
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from test_basix import CELLS, mixed, quadrilaterals, square
from testutil import contains

from uflx_compiler import compile_form, evaluate, fold, runtime_tables
from uflx_oracle import evaluate as oracle
from uflx_spec_interpreter import (
    RD,
    cell_domain,
    domain_map,
    function,
    gradient,
    index,
    index_sum,
    integral,
    normal,
    num,
    pull_back,
    side_map,
)
from uflx_spec_interpreter._src.nodes import Coordinate, Gradient, Integral, PullBack
from uflx_toy.basix_cells import basix_cell
from uflx_toy.basix_spaces import basix_space
from uflx_toy.runtime import runtime_array, runtime_extent

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


def facet_functionals(D, E, B, u):
    plus, minus, inside = side_map(D, E, "+"), side_map(D, E, "-"), side_map(D, B, "inside")
    jump = pull_back(u, plus) - pull_back(u, minus)
    g = gradient(u, D)
    return [
        integral(pull_back(u, plus) * normal(plus)[0], E),
        integral(jump * jump, E),
        integral(pull_back(g[0], inside) * normal(inside)[0], B),
        integral(pull_back(u, minus) * pull_back(g[1], plus), E),
    ]


@pytest.mark.parametrize("mesh", ["triangles", "quadrilaterals", "mixed"])
@pytest.mark.parametrize("discontinuous", [False, True])
def test_compiled_facet_integrals_agree_with_the_oracle(mesh, discontinuous):
    D, E, B = {
        "triangles": lambda: square(basix_cell(CELLS[2])),
        "quadrilaterals": quadrilaterals,
        "mixed": mixed,
    }[mesh]()
    V = basix_space("V", D, P, 1, discontinuous)
    u = function(V, "u")
    U = np.random.default_rng(8).normal(size=V.num_dofs)
    env = {("coefficients", u): U}
    for e in facet_functionals(D, E, B, u):
        compiled = compile_form(e, 4)
        lazy = (Integral, PullBack, Gradient, Coordinate)
        assert not contains(compiled, lambda n, lazy=lazy: isinstance(n, lazy))
        expected = oracle(e, {u: U})(())
        assert np.isclose(
            evaluate(compiled, env, np, fold(compiled)), expected, rtol=1e-8, atol=1e-12
        ), e


def test_a_function_at_points_known_only_at_assembly():
    # The cut cell example on a mesh: each cell has its own points, which the solver gives.
    D = square(basix_cell(CELLS[2]))[0]
    V = basix_space("V", D, P, 2)
    u = function(V, "u")
    K = D.pieces[0].parameter_domain
    ((c, cells),) = D.pieces[0].parameterisation.free_indices
    q, i = runtime_extent("q"), index("i")
    points = runtime_array("points", (cells, q, 2))
    weights = runtime_array("weights", (cells, q))
    xs = domain_map(RD(0), points[c, i], K)
    cell = cell_domain(D, 0)
    rng = np.random.default_rng(11)
    U = rng.normal(size=V.num_dofs)
    for f in (u, gradient(u, D)[0]):
        e = index_sum(
            index_sum(weights[c, i] * integral(pull_back(pull_back(f, cell), xs), RD(0)), (i,)),
            (c,),
        )
        compiled = compile_form(e, 2)
        for n in (1, 3, 6):
            # Points in the reference triangle, as a cut of each cell would give.
            X = rng.dirichlet(np.ones(3), size=(cells, n))[..., 1:]
            W = rng.random((cells, n))
            env = runtime_tables(
                compiled, {q: n, "points": X, "weights": W, ("coefficients", u): U}
            )
            expected = oracle(e, {u: U, q: n, points: X, weights: W})(())
            assert np.isclose(evaluate(compiled, env), expected, rtol=1e-8)
            # JAX takes the arrays in a fixed order, and the extent stays a Python int.
            keys = [k for k in env if k != q]

            def run(arrays, keys=keys, n=n, compiled=compiled):
                return evaluate(compiled, dict(zip(keys, arrays, strict=True)) | {q: n}, jnp)

            arrays = tuple(jnp.asarray(env[k]) for k in keys)
            assert np.isclose(jax.jit(run)(arrays), expected)
