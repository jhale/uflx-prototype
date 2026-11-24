# Copyright (C) 2026 Jack S. Hale
#
# This file is part of uflx-core-experimental (https://fenics.github.io/uflx/language/)
#
# SPDX-License-Identifier:    MIT

import pytest

from uflx.calculus import directional_derivative
from uflx.forms import function
from uflx.nodes import Complex, Integer
from uflx_extensions.complex_differentiation import complex_derivative_parts

from ..uflx.spaces import space


@pytest.mark.parametrize("grouped", [False, True])
def test_projection_expands_into_core_derivatives(grouped):
    V = space("C", scalar_type="complex")
    x, y, a, b, h, k = (function(V, name) for name in "xyabhk")
    i = Complex(Integer(0), Integer(1))
    variables, points, direction = ((x, y), (a, b), (h, k)) if grouped else (x, a, h)
    body = x * y
    delta = directional_derivative(body, variables, points, direction)
    rotated = directional_derivative(body, variables, points, (i * h, i * k) if grouped else i * h)
    plus, minus = complex_derivative_parts(body, variables, points, direction)
    assert plus is (delta - i * rotated) / Integer(2)
    assert minus is (delta + i * rotated) / Integer(2)


def test_projection_requires_complex_variables():
    r = function(space("R", scalar_type="real"), "r")
    with pytest.raises(ValueError, match="complex variable"):
        complex_derivative_parts(r * r, r, r, r)
