# Copyright (C) 2026 Jack S. Hale
#
# This file is part of uflx-core-experimental (https://fenics.github.io/uflx/language/)
#
# SPDX-License-Identifier:    MIT

import numpy as np
import pytest

from uflx._src.nodes import PullBack
from uflx._src.passes import applications, derivatives, gradients, integrals, pullbacks
from uflx._src.passes.contracts import in_core_normal_form
from uflx.arithmetic import num
from uflx.calculus import coordinate, integral
from uflx.geometry import domain_map, pull_back
from uflx.nodes import EuclideanSpace
from uflx.tensors import index_sum
from uflx_jax_compiler import trace

from ..testutil import contains
from ..uflx.test_runtime_extent import i, points, q, weights
from ..uflx_basix.cells import reference_simplex

T, POINT = reference_simplex(2), EuclideanSpace(0)
X = coordinate(T)
f = X[0] * X[0] * X[1] + num(3) * X[0]


def runtime_quadrature():
    # The cut cell example, with q points known only at assembly.
    xs = domain_map(POINT, points[i], T)
    return index_sum(weights[i] * integral(pull_back(f, xs), POINT), (i,))


def test_integer_indices_check_known_bounds_only():
    data = np.array([[1.0, 2.0], [3.0, 4.0]])
    env = {q: 2, points: data}
    np.testing.assert_array_equal(trace(points[0], *env)(*env.values()), data[0])
    assert trace(points[1, 0], *env)(*env.values()) == data[1, 0]
    with pytest.raises(IndexError):
        points[-1, 0]
    with pytest.raises(IndexError):
        points[0, 2]


def test_runtime_quadrature_passes_through_every_pass_unchanged_in_extent():
    e = runtime_quadrature()
    out = e
    for apply in (
        applications.expand_applications,
        derivatives.apply_derivatives,
        integrals.expand_integrals,
        gradients.apply_gradients,
        pullbacks.expand_pull_backs,
        gradients.apply_gradients,
        pullbacks.expand_pull_backs,
    ):
        out = apply(out)
    assert in_core_normal_form(out)
    assert contains(e, lambda n: isinstance(n, PullBack))
    assert not contains(out, lambda n: isinstance(n, PullBack))

    rng = np.random.default_rng(2)
    for n in (1, 4, 7):
        P, w = rng.random((n, 2)), rng.random(n)
        expected = sum(wk * (p[0] ** 2 * p[1] + 3 * p[0]) for p, wk in zip(P, w, strict=True))
        env = {q: n, points: P, weights: w}
        for g in (e, out):
            assert np.isclose(trace(g, *env)(*env.values()), expected)
