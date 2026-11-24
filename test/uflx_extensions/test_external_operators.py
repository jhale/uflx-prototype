# Copyright (C) 2026 Jack S. Hale
#
# This file is part of uflx-core-experimental (https://fenics.github.io/uflx/language/)
#
# SPDX-License-Identifier:    MIT

import pytest

from uflx.arithmetic import num
from uflx.calculus import directional_derivative
from uflx.forms import function
from uflx.tensors import index, tensor
from uflx_extensions.external_operators import apply_external_derivatives, external_operator

from ..uflx.spaces import space
from .operations import Energy, EnergyTangent


def test_derivative_action_receives_points_then_directions():
    V = space("R", None, scalar_type="real")
    x, y, a, b, h, k = (function(V, name) for name in "xyabhk")
    energy = external_operator(Energy(), (x, y))
    derivative = directional_derivative(energy, (x, y), (a, b), (h, k))
    action = apply_external_derivatives(derivative)
    assert action is external_operator(EnergyTangent(), (a, b, h, k))
    combined = directional_derivative(energy + x, (x, y), (a, b), (h, k))
    assert apply_external_derivatives(combined) is action + h
    with pytest.raises(TypeError, match="no derivative action"):
        apply_external_derivatives(directional_derivative(action, a, a, h))


def test_operator_inherits_free_indices_from_its_operands():
    i, j = index("i"), index("j")
    data = tensor((num(2), num(3)))
    operator = external_operator(Energy(), (data[j], data[i]))
    assert operator.free_indices == ((j, 2), (i, 2))
