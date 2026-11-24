# Copyright (C) 2026 Jack S. Hale
#
# This file is part of uflx-core-experimental (https://fenics.github.io/uflx/language/)
#
# SPDX-License-Identifier:    MIT
"""Zero transformations preserve the declared scalar type and other static attributes."""

import pytest

from uflx._src.literals import zero
from uflx.calculus import directional_derivative
from uflx.forms import function
from uflx.nodes import ScalarType
from uflx.passes import apply_derivatives
from uflx.tensors import index

from .spaces import space


@pytest.mark.parametrize("bad", ["invalid", complex, None, 1])
def test_zero_rejects_invalid_scalar_type(bad):
    with pytest.raises(TypeError):
        zero((), (), (), bad)


@pytest.mark.parametrize("scalar_type", (ScalarType.real, ScalarType.complex))
def test_constant_derivative_preserves_output_type_not_direction_type(scalar_type):
    k = index("derivative-family")
    V = space("complex-variable")
    u = function(V, "u")
    h = function(V, "h")
    z = zero((2,), ((k, 3),), (), scalar_type)
    out = apply_derivatives(directional_derivative(z, u, u, h))
    assert out is z
