"""Exact identity checks shared by reconstruction and alpha comparison."""

import pytest

from uflx._src.algorithms.binding import alpha_equivalent
from uflx._src.algorithms.traversal import rebuild
from uflx._src.nodes.base import _encode, construction_equal
from uflx.forms import form, function
from uflx.tensors import index
from uflx_basix.spaces import space


@pytest.mark.parametrize(
    ("left", "right"),
    [
        (1, 1.0),
        (0.0, -0.0),
        (("label", (1,)), ("label", (1.0,))),
        (("label", 0.0), ("label", -0.0)),
        (("label", 2.5), ("label", 2.5)),
        ((1,), (1, 2)),
        (index("i"), index("j")),
        (index("i"), index("i")),
    ],
)
def test_agrees_with_construction_encoding(left, right):
    assert construction_equal(left, right) == (_encode(left) == _encode(right))
    assert construction_equal(right, left) == construction_equal(left, right)


@pytest.mark.parametrize(("first", "second"), [(1, 1.0), (0.0, -0.0)])
def test_rebuild_does_not_discard_identity_changes(first, second):
    V = space("construction-equality", scalar_type="real")
    original = function(V, ("label", first))
    expected = function(V, ("label", second))
    assert original is not expected
    assert rebuild(original, label=("label", second)) is expected
    assert rebuild(original, label=("label", first)) is original


def test_embedded_nodes_are_not_alpha_normalised():
    V = space("opaque-construction", scalar_type="real")
    u, v = function(V, "u"), function(V, "v")
    a, b = form(u, (u,)), form(v, (v,))
    assert alpha_equivalent(a, b)
    assert not construction_equal(("label", a), ("label", b))


def test_deep_tuples_use_a_work_stack():
    left, same, different = 1, 1, 1.0
    for _ in range(2000):
        left, same, different = (left,), (same,), (different,)
    assert construction_equal(left, same)
    assert not construction_equal(left, different)
