# Copyright (C) 2026 Jack S. Hale
#
# This file is part of uflx-core-experimental (https://fenics.github.io/uflx/language/)
#
# SPDX-License-Identifier:    MIT
"""Regressions for rewrite completion, extension boundaries and value substitution."""

from __future__ import annotations

import numpy as np

from uflx.arithmetic import num
from uflx.calculus import directional_derivative
from uflx.forms import function
from uflx.passes import apply_derivatives
from uflx_jax_compiler import trace

from ..uflx.spaces import space


def test_at_point_derivative_does_not_substitute_into_inserted_direction():
    x = function(space("direct-derivative"), "x")
    e = directional_derivative(x * x, x, num(2), x)
    out = apply_derivatives(e)
    assert np.isclose(trace(out, x)(3), 12)
