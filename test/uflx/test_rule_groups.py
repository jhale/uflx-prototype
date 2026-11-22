"""Rule precedence and type completion at mathematical group boundaries."""

from uflx._src.algorithms.rewriting import complete_indices
from uflx.nodes import Zero
from uflx.tensors import index


def test_zero_completion_keeps_a_typed_zero():
    i = index("i")
    expected = Zero((2,), ((i, 3),), (), "complex")
    assert complete_indices(expected, Zero((2,), (), (), "real")) is expected
