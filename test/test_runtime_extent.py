import numpy as np
from testutil import contains

from uflx_spec_interpreter.calculus import coordinate, integral
from uflx_spec_interpreter.domains import coordinate_domain
from uflx_spec_interpreter.extensions.runtime import runtime_array, runtime_extent
from uflx_spec_interpreter.indices import index
from uflx_spec_interpreter.literals import num
from uflx_spec_interpreter.maps import domain_map, pull_back
from uflx_spec_interpreter.nodes import PullBack
from uflx_spec_interpreter.oracle import evaluate
from uflx_spec_interpreter.passes import (
    actions,
    derivatives,
    gateaux,
    gradients,
    integrals,
    pullbacks,
)
from uflx_spec_interpreter.tensors import index_sum

i = index("i")
q, r = runtime_extent("q"), runtime_extent("r")
T, POINT = coordinate_domain("T", 2), coordinate_domain("pt", 0)
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
        gateaux.apply_gateaux_derivatives,
        actions.apply_actions,
        derivatives.apply_derivatives,
        integrals.expand_integrals,
        pullbacks.expand_pull_backs,
        gradients.apply_gradients,
    ):
        out = apply(out)
    assert gradients.in_n6(out)
    assert contains(e, lambda n: isinstance(n, PullBack))
    assert not contains(out, lambda n: isinstance(n, PullBack))

    rng = np.random.default_rng(2)
    for n in (1, 4, 7):
        P, w = rng.random((n, 2)), rng.random(n)
        expected = sum(wk * (p[0] ** 2 * p[1] + 3 * p[0]) for p, wk in zip(P, w, strict=True))
        env = {q: n, points: P, weights: w}
        for g in (e, out):
            assert np.isclose(evaluate(g, env)(()), expected)
