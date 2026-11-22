"""Zero signatures and complex tensor-valued derivative forms."""

import numpy as np
import pytest

from uflx._src.extensions.complex_differentiation import complex_derivative_parts
from uflx._src.nodes import AbstractExpression, Zero
from uflx.arithmetic import conjugate, num
from uflx.calculus import directional_derivative
from uflx.forms import apply, form, function
from uflx.passes import apply_derivatives
from uflx.tensors import tensor
from uflx_basix.spaces import space
from uflx_oracle import evaluate


class ExternalValue(AbstractExpression):
    """Opaque operator with exposed dependence but no supplied derivative rule."""

    identifier = "reproduction355.ExternalValue"
    value_fields = ("operand",)
    operand: AbstractExpression

    @property
    def value_shape(self):
        return self.operand.value_shape

    @property
    def free_indices(self):
        return self.operand.free_indices

    @property
    def argument_spaces(self):
        return self.operand.argument_spaces

    @property
    def scalar_type(self):
        return self.operand.scalar_type

    @property
    def domain(self):
        return self.operand.domain


@pytest.mark.parametrize("shape", [(), (2,)])
@pytest.mark.parametrize("scalar_type", ["real", "complex"])
@pytest.mark.parametrize("existing_slot", [False, True])
@pytest.mark.parametrize("derivative_kind", ["directional", "form"])
def test_independent_external_operator_derivative(
    shape, scalar_type, existing_slot, derivative_kind
):
    V = space("pr355", value_shape=shape, scalar_type=scalar_type)
    S = space("pr355-scalar", scalar_type=scalar_type)
    u, coefficient, h, k = (function(V, name) for name in ("u", "coefficient", "h", "k"))
    v = function(S, "v")
    operand = form(coefficient * v, (v,)) if existing_slot else coefficient
    operator = ExternalValue(operand)
    slots = (S,) if existing_slot else ()

    zero = apply_derivatives(directional_derivative(operator, u, u, h))
    assert isinstance(zero, Zero)
    assert zero.value_shape == shape
    assert zero.argument_spaces == slots
    assert zero.scalar_type == scalar_type
    assert zero.free_indices == ()
    if derivative_kind == "directional":
        return

    derivative = apply_derivatives(form(directional_derivative(operator, u, u, h), (h,)))
    assert derivative.value_shape == shape
    assert derivative.argument_spaces == (*slots, V)
    assert derivative.scalar_type == scalar_type
    direction = np.ones(shape) if shape else 1.0
    arguments = ((2.0,) if existing_slot else ()) + (direction,)
    np.testing.assert_array_equal(evaluate(derivative, {})(arguments), np.zeros(shape))

    # The zero remains usable as a form, including for another derivative.
    contracted = apply_derivatives(
        form(apply(derivative, (v, h)), (v,)) if existing_slot else apply(derivative, (h,))
    )
    assert contracted.argument_spaces == slots
    second = apply_derivatives(form(directional_derivative(derivative, u, u, k), (k,)))
    assert second.argument_spaces == (*slots, V, V)
    np.testing.assert_array_equal(evaluate(second, {})((*arguments, direction)), np.zeros(shape))


@pytest.mark.parametrize("shape", [(2,), (2, 2)])
@pytest.mark.parametrize("antilinear", [False, True])
def test_nonzero_tensor_wirtinger_parts(shape, antilinear):
    V = space("tensor-wirtinger", value_shape=shape, scalar_type="complex")
    u, h = function(V, "u"), function(V, "h")
    expression = num(2) * u + num(3) * conjugate(u)
    entry = conjugate(h) if antilinear else h
    derivative = apply_derivatives(
        form(complex_derivative_parts(expression, u, u, h)[int(antilinear)], (entry,))
    )
    assert derivative.value_shape == shape
    direction = np.arange(np.prod(shape)).reshape(shape) + (0.7 + 0.4j)
    expected = (3 if antilinear else 2) * direction
    np.testing.assert_allclose(
        evaluate(derivative, {u: direction * (1 - 2j)})((direction,)), expected
    )


def test_vector_action_derivative_with_both_operands_dependent():
    V = space("vector-action", scalar_type="real")
    u, v, h = (function(V, name) for name in ("u", "v", "h"))
    operator = form(tensor((u * v, u * u * v)), (v,))
    expression = apply(operator, (u,))
    derivative = apply_derivatives(directional_derivative(expression, u, u, h))
    assert derivative.value_shape == (2,)
    np.testing.assert_allclose(evaluate(derivative, {u: 2.0, h: 0.5})(()), [2.0, 6.0])
