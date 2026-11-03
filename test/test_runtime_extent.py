import numpy as np
from testutil import contains

from uflx_oracle import evaluate
from uflx_spec_interpreter import (
    RD,
    coordinate,
    domain_map,
    index,
    index_sum,
    integral,
    num,
    pull_back,
)
from uflx_spec_interpreter._src.nodes import PullBack
from uflx_spec_interpreter._src.passes import (
    actions,
    derivatives,
    gradients,
    integrals,
    pullbacks,
)
from uflx_toy.cells import reference_simplex
from uflx_toy.runtime import runtime_array, runtime_extent

i = index("i")
q, r = runtime_extent("q"), runtime_extent("r")
T, POINT = reference_simplex(2), RD(0)
points, weights = runtime_array("points", (q, 2)), runtime_array("weights", (q,))
X = coordinate(T)
f = X[0] * X[0] * X[1] + num(3) * X[0]


def runtime_quadrature():
    # The cut cell example, with q points known only at assembly.
    xs = domain_map(POINT, points[i], T)
    return index_sum(weights[i] * integral(pull_back(f, xs), POINT), (i,))


def test_an_abstract_extent_is_compared_by_equality():
    assert runtime_extent("q") == q != r
    assert points.value_shape == (q, 2)
    assert dict((weights[i] * points[i, 0]).free_indices) == {i: q}
    assert points[i].value_shape == (2,)


def test_runtime_quadrature_passes_through_every_pass_unchanged_in_extent():
    e = runtime_quadrature()
    out = e
    for apply in (
        actions.apply_actions,
        derivatives.apply_derivatives,
        integrals.expand_integrals,
        gradients.apply_gradients,
        pullbacks.expand_pull_backs,
        gradients.apply_gradients,
        pullbacks.expand_pull_backs,
    ):
        out = apply(out)
    assert gradients._in_n6(out)
    assert contains(e, lambda n: isinstance(n, PullBack))
    assert not contains(out, lambda n: isinstance(n, PullBack))

    rng = np.random.default_rng(2)
    for n in (1, 4, 7):
        P, w = rng.random((n, 2)), rng.random(n)
        expected = sum(wk * (p[0] ** 2 * p[1] + 3 * p[0]) for p, wk in zip(P, w, strict=True))
        env = {q: n, points: P, weights: w}
        for g in (e, out):
            assert np.isclose(evaluate(g, env)(()), expected)


def test_the_compiler_reads_the_extent_and_the_points_at_assembly():
    import jax
    import jax.numpy as jnp

    from uflx_compiler import compile_form
    from uflx_compiler import evaluate as run

    jax.config.update("jax_enable_x64", True)
    compiled = compile_form(runtime_quadrature(), 2)
    for n in (1, 4, 7):
        rng = np.random.default_rng(n)
        P, w = rng.random((n, 2)), rng.random(n)
        expected = sum(wk * (p[0] ** 2 * p[1] + 3 * p[0]) for p, wk in zip(P, w, strict=True))
        env = {"q": n, points.label: P, weights.label: w}
        assert np.isclose(run(compiled, env), expected)
        f = jax.jit(
            lambda P, w, n=n: run(compiled, {"q": n, points.label: P, weights.label: w}, jnp)
        )
        assert np.isclose(f(P, w), expected)
