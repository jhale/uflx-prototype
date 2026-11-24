# Copyright (C) 2026 Jack S. Hale
#
# This file is part of uflx-core-experimental (https://fenics.github.io/uflx/language/)
#
# SPDX-License-Identifier:    MIT
"""Completed recipes honor output contracts after substitution and extension work."""

from uflx.arithmetic import num
from uflx.calculus import directional_derivative
from uflx.forms import function
from uflx.passes import lower

from .spaces import space


def test_lowering_is_idempotent():
    x = function(space("lower-idempotent"), "x")
    out = lower(directional_derivative(x * x, x, x, num(1)))
    assert lower(out) is out
