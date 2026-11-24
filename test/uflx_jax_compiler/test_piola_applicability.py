# Copyright (C) 2026 Jack S. Hale
#
# This file is part of uflx-core-experimental (https://fenics.github.io/uflx/language/)
#
# SPDX-License-Identifier:    MIT
"""One applicability decision is shared by gradients, pull-backs and contracts."""

import numpy as np

from uflx._src.algorithms.traversal import everywhere
from uflx._src.nodes import Curl, Divergence, PullBack
from uflx.calculus import coordinate, curl, divergence
from uflx.nodes import EuclideanSpace
from uflx.passes import apply_gradients
from uflx.tensors import tensor
from uflx_jax_compiler import trace


def test_ordinary_divergence_and_curl_expand_without_any_pullback():
    P = EuclideanSpace(2)
    x, y = coordinate(P)[0], coordinate(P)[1]
    cases = (
        (divergence(tensor((x * x, y * y)), P), 2 * 0.3 + 2 * 0.7),
        (curl(tensor((x * y, x * x)), P), 0.3),
    )
    for original, expected in cases:
        out = apply_gradients(original)
        assert everywhere(out, lambda node: not isinstance(node, Divergence | Curl | PullBack))
        actual = trace(out, P)(np.array([0.3, 0.7]))
        np.testing.assert_allclose(actual, expected)
