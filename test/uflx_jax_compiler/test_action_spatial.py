# Copyright (C) 2026 Jack S. Hale
#
# This file is part of uflx-core-experimental (https://fenics.github.io/uflx/language/)
#
# SPDX-License-Identifier:    MIT
"""Spatial operators transform a form's output without transforming its arguments."""

from __future__ import annotations

import numpy as np
import pytest

from uflx._src.nodes import Apply
from uflx._src.passes.contracts import core_normal_form
from uflx._src.passes.lowering import lower_spatial
from uflx.calculus import gradient
from uflx.forms import apply
from uflx.geometry import pull_back
from uflx.passes import apply_gradients, expand_pull_backs, lower

from ..uflx_basix.test_action_spatial import setup
from .reference import reference_trace


def value(e, domain, point):
    return np.asarray(reference_trace(e, domain)(np.array([point]))).real


def test_pullback_keeps_the_original_field_argument_and_value():
    P, K, x, _, A, psi = setup()
    g = x * x
    e = pull_back(apply(gradient(A, P), (g,)), psi)
    out = expand_pull_backs(e)
    assert isinstance(out, Apply)
    assert out.arguments[0] == g and out.arguments[0].domain is P
    assert out.argument_spaces == e.argument_spaces
    np.testing.assert_allclose(value(out, K, 0.3), [3 * 1.6**2], atol=1e-6)
    np.testing.assert_allclose(value(out, K, 0.3), value(e, K, 0.3), atol=1e-6)
    complete = lower(e)
    assert not core_normal_form(complete)
    assert lower_spatial(complete) is complete
    np.testing.assert_allclose(value(complete, K, 0.3), [3 * 1.6**2], atol=1e-5)


@pytest.mark.parametrize("retained", [False, True])
def test_gradient_does_not_count_argument_variation_twice(retained):
    P, _, x, _, A, _ = setup()
    f = gradient(A, P) if retained else A
    e = gradient(apply(f, (x * x,)), P)
    out = apply_gradients(e)
    expected = [[6 * 1.6]] if retained else [3 * 1.6**2]
    np.testing.assert_allclose(value(out, P, 1.6), expected, atol=1e-5)
    complete = lower_spatial(out)
    np.testing.assert_allclose(value(complete, P, 1.6), expected, atol=1e-5)
    assert lower_spatial(complete) is complete
