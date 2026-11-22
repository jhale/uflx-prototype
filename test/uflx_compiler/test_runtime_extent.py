import numpy as np

from ..uflx_basix.test_runtime_extent import points, weights
from ..uflx_oracle.test_runtime_extent import runtime_quadrature


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
