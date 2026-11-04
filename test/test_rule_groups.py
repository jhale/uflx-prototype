"""Rule precedence and type completion at mathematical group boundaries."""

import numpy as np
import pytest

from uflx_compiler import evaluate
from uflx_oracle import evaluate as oracle
from uflx_spec_interpreter import (
    RD,
    Zero,
    apply_derivatives,
    apply_gradients,
    component_tensor,
    coordinate,
    directional_derivative,
    function,
    gradient,
    index,
    index_sum,
    num,
    power,
    tensor,
)
from uflx_spec_interpreter._src.algorithms.rewriting import _complete_indices
from uflx_toy.spaces import space


@pytest.mark.parametrize(
    "exponent",
    [num(0), num(-0.0), num(0j), Zero((), (), (), "real"), num(1), num(1.0), num(1 + 0j)],
)
def test_literal_power_rules_at_zero_base(exponent):
    u = function(space("powers", scalar_type="real"), "u")
    e = directional_derivative(power(u, exponent), u, num(0), num(3))
    out = apply_derivatives(e)
    with np.errstate(all="raise"):
        expected = 0 if isinstance(exponent, Zero) or oracle(exponent, {})(()) == 0 else 3
        assert evaluate(out, {}) == expected
    assert out.scalar_type == e.scalar_type


@pytest.mark.parametrize("exponent", [num(0), num(0j), num(1), num(1 + 0j)])
def test_spatial_power_rules_preserve_the_component_axis(exponent):
    P = RD(2)
    e = gradient(power(coordinate(P)[0], exponent), P)
    out = apply_gradients(e)
    expected = [0, 0] if oracle(exponent, {})(()) == 0 else [1, 0]
    np.testing.assert_array_equal(evaluate(out, {}), expected)
    assert out.value_shape == (2,)
    assert out.scalar_type == e.scalar_type


def test_variable_exponent_at_one_still_has_its_derivative():
    u = function(space("variable-power", scalar_type="real"), "u")
    out = apply_derivatives(directional_derivative(power(num(2), u), u, num(1), num(1)))
    assert np.isclose(evaluate(out, {}), 2 * np.log(2))


def test_power_type_completion_before_rebuilding_an_index_binder():
    u = function(space("bound-power", scalar_type="real"), "u")
    i = index("i")
    # The exponent is literally zero but carries a family index.
    e = power(u, Zero((), ((i, 3),), (), "real"))
    out = apply_derivatives(component_tensor(directional_derivative(e, u, num(0), num(1)), (i,)))
    np.testing.assert_array_equal(evaluate(out, {}), [0, 0, 0])


def test_capture_avoidance_across_structural_and_derivative_rules():
    u = function(space("capture", scalar_type="real"), "u")
    i = index("i")
    a = tensor((num(2), num(3)))
    b = tensor((num(7), num(11)))
    e = directional_derivative(index_sum(u * a[i], (i,)), u, num(0), b[i])
    out = apply_derivatives(component_tensor(e, (i,)))
    np.testing.assert_array_equal(evaluate(out, {}), [35, 55])


def test_zero_completion_keeps_a_typed_zero():
    i = index("i")
    expected = Zero((2,), ((i, 3),), (), "complex")
    assert _complete_indices(expected, Zero((2,), (), (), "real")) is expected
