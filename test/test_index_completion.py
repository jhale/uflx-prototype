"""Family indices survive rewrites before an enclosing index binder is rebuilt."""

import pytest

from uflx_spec_interpreter import (
    apply_derivatives,
    component_tensor,
    directional_derivative,
    function,
    index,
    num,
    tensor,
)
from uflx_spec_interpreter._src.nodes import Zero
from uflx_spec_interpreter._src.static import complete_indices
from uflx_toy.spaces import space


def test_derivative_restores_unused_point_indices_before_rebuilding_parent():
    u = function(space("family-point"), "u")
    k = index("k")
    a = tensor((num(2), num(3)))[k]
    derivative = directional_derivative(u, u, a, num(1))
    out = apply_derivatives(component_tensor(derivative, (k,)))
    assert out.value_shape == (2,)
    assert out.free_indices == ()
    assert out.expression.free_indices == ((k, 2),)


def test_completion_is_idempotent_and_accepts_reordered_indices():
    i, j = index("i"), index("j")
    original = Zero((), ((i, 2), (j, 3)), ())
    out = complete_indices(original, num(1))
    assert out.free_indices == original.free_indices
    assert complete_indices(original, out) is out
    reordered = Zero((), ((j, 3), (i, 2)), ())
    assert complete_indices(original, reordered) is reordered


@pytest.mark.parametrize("kind", ["extra", "extent", "shape", "arity"])
def test_completion_rejects_incompatible_replacements(kind):
    i, j = index("i"), index("j")
    original = Zero((), ((i, 2),), ())
    replacement = {
        "extra": Zero((), ((j, 2),), ()),
        "extent": Zero((), ((i, 3),), ()),
        "shape": Zero((2,), ((i, 2),), ()),
        "arity": Zero((), ((i, 2),), (space("slot"),)),
    }[kind]
    with pytest.raises(ValueError):
        complete_indices(original, replacement)
