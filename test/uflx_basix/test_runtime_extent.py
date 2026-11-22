from uflx.tensors import index
from uflx_basix.runtime import runtime_array, runtime_extent

i = index("i")
q, r = runtime_extent("q"), runtime_extent("r")
points, weights = runtime_array("points", (q, 2)), runtime_array("weights", (q,))


def test_an_abstract_extent_is_compared_by_equality():
    assert runtime_extent("q") == q != r
    assert points.value_shape == (q, 2)
    assert dict((weights[i] * points[i, 0]).free_indices) == {i: q}
    assert points[i].value_shape == (2,)
