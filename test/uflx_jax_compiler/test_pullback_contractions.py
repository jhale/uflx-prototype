# Copyright (C) 2026 Jack S. Hale
#
# This file is part of uflx-core-experimental (https://fenics.github.io/uflx/language/)
#
# SPDX-License-Identifier:    MIT
"""Shared transformation laws preserve canonical cell geometry and gradient values."""

import jax.numpy as jnp
import numpy as np
import pytest

from uflx._src.nodes import Gradient, TransformationLaw
from uflx._src.passes.lowering import lower_spatial
from uflx._src.rules.coordinates import transformed
from uflx.calculus import coordinate, gradient
from uflx.forms import function
from uflx.geometry import cell_map, pull_back
from uflx.passes import expand_pull_backs
from uflx_jax_compiler import trace

from ..uflx.spaces import space
from ..uflx_basix.test_pullback_contractions import geometry


@pytest.mark.parametrize("vector", [False, True])
def test_gradient_pullback_uses_covariant_law_for_scalar_and_vector_fields(vector):
    _, x, _, D = geometry()
    u = function(space("physical-field", value_shape=(2,) if vector else (), domain=D), "u")
    C = cell_map(D, 0)
    original = pull_back(gradient(u, D), C)
    actual = expand_pull_backs(original)
    expected = transformed(
        Gradient(expand_pull_backs(pull_back(u, C)), C.reference_cell),
        TransformationLaw(("covariant",), 0),
        C,
    )
    assert actual is expected
    assert actual.value_shape == original.value_shape
    assert actual.free_indices == original.free_indices
    assert actual.argument_spaces == original.argument_spaces
    assert actual.domain is original.domain
    # Straight curve (t, 2t) in R^2: the tangent projection of (1,0) is (1,2)/5.
    # Component two, for a vector field, has ambient gradient (0,1).
    env = {x: lambda p: jnp.array([p[0], 2 * p[0]])}
    env[u] = coordinate(D) if vector else coordinate(D)[0]
    out = lower_spatial(actual)
    value = trace(out, *env, C.reference_cell)(*env.values(), [0.3])
    desired = np.array([[1, 2], [2, 4]]) / 5 if vector else np.array([1, 2]) / 5
    np.testing.assert_allclose(value, desired, atol=1e-6)
