"""Regression examples in the order of the revised language chapters."""

from __future__ import annotations

import pytest

from uflx.nodes import Integer
from uflx.tensors import index, index_sum
from uflx_basix.runtime import runtime_array, runtime_extent
from uflx_oracle.interpreter import Interpreter


@pytest.mark.parametrize("symbolic", [False, True])
def test_empty_sum_does_not_read_the_summand(symbolic):
    from uflx_compiler import evaluate as compiled_value

    n = runtime_extent("empty") if symbolic else 0
    i = index("empty-index")
    f = runtime_array("unavailable", (n,))
    total = index_sum(Integer(2) * f[i], (i,))
    env = {"empty": 0, n: 0} if symbolic else {}
    assert Interpreter(env).evaluate(total) == 0
    assert compiled_value(total, env) == 0
