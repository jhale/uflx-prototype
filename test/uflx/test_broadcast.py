"""Broadcasting preserves value axes, family indices and lazy empty values."""

import numpy as np
import pytest

from uflx._src.algorithms.rewriting import complete_indices
from uflx._src.nodes import BroadcastTo
from uflx.algorithms import replace
from uflx.arithmetic import num, square_root
from uflx.calculus import coordinate, directional_derivative, gradient
from uflx.forms import apply, form, function
from uflx.geometry import domain_map, pull_back
from uflx.nodes import EuclideanSpace, Zero
from uflx.passes import (
    apply_derivatives,
    apply_gradients,
    expand_applications,
    expand_pull_backs,
    lower,
)
from uflx.tensors import broadcast_to, component_tensor, index, index_sum, tensor
from uflx_basix.runtime import runtime_array, runtime_extent
from uflx_basix.spaces import space
from uflx_compiler import evaluate
from uflx_oracle import evaluate as oracle


@pytest.mark.parametrize(
    "source,target",
    [
        ((), ()),
        ((), (2, 3)),
        ((3,), (2, 3)),
        ((2, 1), (4, 2, 3)),
        ((1,), (0,)),
        ((0,), (2, 0)),
    ],
)
def test_values(source, target):
    x = runtime_array("x", source)
    data = np.arange(np.prod(source)).reshape(source)
    e = broadcast_to(x, target)
    expected = np.broadcast_to(data, target)
    np.testing.assert_array_equal(evaluate(e, {"x": data}), expected)
    np.testing.assert_array_equal(oracle(e, {x: data})(()), expected)
    assert lower(e) is e


@pytest.mark.parametrize("source,target", [((2,), ()), ((2,), (3,)), ((0,), (1,)), ((), (-1,))])
def test_invalid_shapes(source, target):
    with pytest.raises(ValueError):
        broadcast_to(Zero(source, (), (), "real"), target)


def test_symbolic_extents():
    m, n = runtime_extent("m"), runtime_extent("n")
    x = runtime_array("x", (m,))
    assert broadcast_to(x, (2, m)).value_shape == (2, m)
    with pytest.raises(ValueError):
        broadcast_to(x, (n,))
    e = broadcast_to(num(3), (n,))
    for count in (0, 3):
        np.testing.assert_array_equal(evaluate(e, {"n": count}), np.full(count, 3))
        np.testing.assert_array_equal(oracle(e, {n: count})(()), np.full(count, 3))


def test_empty_target_does_not_read_its_operand():
    missing = runtime_array("missing", ())
    for x in (missing, square_root(num(-1))):
        e = broadcast_to(x, (0, 2))
        with np.errstate(invalid="raise"):
            assert evaluate(e, {}).shape == (0, 2)
            assert oracle(e, {})(()).shape == (0, 2)
        assert replace(e, {x: num(7)}).value is num(7)


def test_broadcast_with_free_indices_keeps_value_axes_separate():
    i, j = index("i"), index("j")
    a = tensor((num(2), num(3)))
    e = broadcast_to(a[i], (3, 2))
    np.testing.assert_array_equal(evaluate(e, {}), np.array([np.full((3, 2), k) for k in (2, 3)]))
    selected = component_tensor(e[j, 1], (i, j))
    np.testing.assert_array_equal(evaluate(selected, {}), [[2, 2, 2], [3, 3, 3]])
    np.testing.assert_array_equal(oracle(selected, {})(()), [[2, 2, 2], [3, 3, 3]])


def test_completion_of_tensor_and_two_missing_indices():
    i, j = index("i"), index("j")
    original = Zero((2,), ((i, 3), (j, 4)), (), "real")
    out = complete_indices(original, tensor((num(5), num(7))))
    assert out.value_shape == (2,)
    assert out.free_indices == original.free_indices
    assert complete_indices(original, out) is out
    np.testing.assert_array_equal(evaluate(out, {}), np.broadcast_to([5, 7], (3, 4, 2)))
    scalar = complete_indices(Zero((), ((i, 3), (j, 0)), (), "real"), num(1))
    assert evaluate(index_sum(scalar, (i, j)), {}) == 0


def test_derivative_completion_uses_broadcast_selection():
    u = function(space("u"), "u")
    i = index("i")
    point = tensor((num(2), num(3)))[i]
    out = apply_derivatives(directional_derivative(u, u, point, num(1)))
    assert isinstance(out.expression, BroadcastTo)
    np.testing.assert_array_equal(evaluate(out, {}), [1, 1])


def test_broadcast_commutes_with_actions_derivatives_and_pullbacks():
    u = function(space("u"), "u")
    e = broadcast_to(u * u, (2, 3))
    d = apply_derivatives(directional_derivative(e, u, num(3), num(1)))
    np.testing.assert_array_equal(evaluate(d, {}), np.full((2, 3), 6))
    a = expand_applications(apply(broadcast_to(form(u, (u,)), (2,)), (num(7),)))
    np.testing.assert_array_equal(evaluate(a, {}), [7, 7])
    P = EuclideanSpace(2)
    x = coordinate(P)
    g = apply_gradients(gradient(broadcast_to(x[0], (3,)), P))
    np.testing.assert_array_equal(evaluate(g, {}), [[1, 0]] * 3)
    mapping = domain_map(EuclideanSpace(0), tensor((num(4), num(5))), P)
    b = expand_pull_backs(pull_back(broadcast_to(x[0], (3,)), mapping))
    np.testing.assert_array_equal(evaluate(b, {}), [4, 4, 4])
