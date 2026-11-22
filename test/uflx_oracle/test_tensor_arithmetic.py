"""Tensor arithmetic stays explicit until scalar component lowering."""

import numpy as np
import pytest

from uflx._src.algorithms.traversal import everywhere
from uflx._src.nodes import Division, Product, Sum
from uflx._src.passes.contracts import in_core_normal_form
from uflx.arithmetic import num
from uflx.calculus import directional_derivative
from uflx.forms import apply, form, function
from uflx.passes import apply_derivatives, expand_arithmetic, lower
from uflx.tensors import index, indexed
from uflx_basix.runtime import runtime_array, runtime_extent
from uflx_basix.spaces import space
from uflx_oracle.interpreter import Interpreter


@pytest.mark.parametrize("shape", [(3,), (2, 3), (0,)])
def test_arithmetic_preserves_nodes_and_lowers_values(shape):
    a = runtime_array("a", shape)
    b = runtime_array("b", shape)
    values = np.arange(np.prod(shape)).reshape(shape) + 1.0
    environment = {a: values, b: 2 * values}
    expressions = (a + b, num(3) * a, a * num(3), a / num(2))
    for expression, node in zip(expressions, (Sum, Product, Product, Division), strict=True):
        assert isinstance(expression, node)
        assert expression.value_shape == shape
        assert not in_core_normal_form(expression)
        expanded = expand_arithmetic(expression)
        assert expanded.value_shape == shape
        assert everywhere(
            expanded,
            lambda e: not isinstance(e, Sum | Product | Division) or not e.value_shape,
        )
        oracle = Interpreter(environment)
        np.testing.assert_allclose(oracle.evaluate(expanded), oracle.evaluate(expression))
        assert expand_arithmetic(expanded) is expanded
        assert lower(expression) is expanded
        assert lower(expanded) is expanded


def test_symbolic_axes_and_free_indices_survive_expansion():
    n = runtime_extent("n")
    a = runtime_array("a", (n,))
    coefficients = runtime_array("coefficients", (2,))
    i = index("i")
    expression = indexed(coefficients, (i,)) * a
    expanded = expand_arithmetic(expression)
    assert expanded.value_shape == (n,)
    assert expanded.free_indices == ((i, 2),)
    for entry in range(2):
        environment = {
            n: 3,
            a: np.array([1.0, 2.0, 3.0]),
            coefficients: np.array([2.0, 4.0]),
            i: entry,
        }
        oracle = Interpreter(environment)
        np.testing.assert_allclose(
            oracle.evaluate(expanded, indices={i: entry}),
            oracle.evaluate(expression, indices={i: entry}),
        )


def test_tensor_product_derivative_and_action():
    x = function(space("scalar", scalar_type="real"), "x")
    v = function(space("vector", (2,), scalar_type="real"), "v")
    expression = x * v
    derivative = apply_derivatives(directional_derivative(expression, x, num(2), num(1)))
    values = {v: np.array([3.0, 5.0])}
    oracle = Interpreter(values)
    np.testing.assert_allclose(oracle.evaluate(lower(derivative)), values[v])
    applied = apply(form(expression, (x,)), (num(4),))
    np.testing.assert_allclose(oracle.evaluate(lower(applied)), 4 * values[v])
