# Copyright (C) 2026 Jack S. Hale
#
# This file is part of uflx-core-experimental (https://fenics.github.io/uflx/language/)
#
# SPDX-License-Identifier:    MIT

import pytest

from uflx.arithmetic import num
from uflx.calculus import coordinate, directional_derivative
from uflx.forms import function
from uflx.geometry import euclidean_space
from uflx.tensors import index, tensor
from uflx_extensions.external_operators import (
    OutputType,
    apply_external_derivatives,
    external_operator,
    operation,
)

from ..uflx.spaces import space
from .operations import Energy, EnergyTangent


def test_operation_signature():
    def preserve(operands):
        (x,) = operands
        return OutputType(x.value_shape, x.scalar_type, x.domain)

    def norm_type(operands):
        (x,) = operands
        if len(x.value_shape) != 1:
            raise ValueError("expected a vector")
        return OutputType((), "real", x.domain)

    law = operation("square", 1, preserve)
    assert law is operation("square", arity=1, output_type=preserve)
    domain = euclidean_space(2)
    x = coordinate(domain)
    assert external_operator(law, (x,)).output_type == ((2,), "real", domain)
    norm = operation("norm", 1, norm_type)
    assert external_operator(norm, (x,)).output_type == ((), "real", domain)
    with pytest.raises(ValueError, match="expects 1 operands"):
        external_operator(law, (x, x))
    with pytest.raises(ValueError, match="expected a vector"):
        external_operator(norm, (x[0],))
    with pytest.raises(ValueError, match="different output type rule"):
        operation("square", 1, norm_type)


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
