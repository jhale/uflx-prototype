"""Tensor arithmetic stays explicit until scalar component lowering."""

import pytest

from uflx_basix.runtime import runtime_array


@pytest.mark.parametrize("operation", [lambda a, b: a + b, lambda a, b: a * b, lambda a, b: a / b])
def test_invalid_tensor_operands(operation):
    with pytest.raises(ValueError):
        operation(runtime_array("a", (2,)), runtime_array("b", (3,)))
