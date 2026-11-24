# Copyright (C) 2026 Jack S. Hale
#
# This file is part of uflx-core-experimental (https://fenics.github.io/uflx/language/)
#
# SPDX-License-Identifier:    MIT
"""Tensor arithmetic stays explicit until scalar component lowering."""

import pytest

from .library import runtime_array


@pytest.mark.parametrize("operation", [lambda a, b: a + b, lambda a, b: a * b, lambda a, b: a / b])
def test_invalid_tensor_operands(operation):
    with pytest.raises(ValueError):
        operation(runtime_array("a", (2,)), runtime_array("b", (3,)))
