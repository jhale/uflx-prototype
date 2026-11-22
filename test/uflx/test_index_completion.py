"""Family indices survive rewrites before an enclosing index binder is rebuilt."""

import pytest

from uflx._src.algorithms.rewriting import complete_indices
from uflx._src.nodes import Zero
from uflx._src.nodes.scalar_types import ScalarType
from uflx.arithmetic import num
from uflx.calculus import directional_derivative
from uflx.forms import function
from uflx.passes import apply_derivatives
from uflx.tensors import component_tensor, index, tensor

from .spaces import space


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
    original = Zero((), ((i, 2), (j, 3)), (), ScalarType.real)
    out = complete_indices(original, num(1))
    assert out.free_indices == original.free_indices
    assert complete_indices(original, out) is out
    reordered = Zero((), ((j, 3), (i, 2)), (), ScalarType.real)
    assert complete_indices(original, reordered) is reordered


@pytest.mark.parametrize("kind", ["extra", "extent", "shape", "arity"])
def test_completion_rejects_incompatible_replacements(kind):
    i, j = index("i"), index("j")
    original = Zero((), ((i, 2),), (), ScalarType.real)
    replacement = {
        "extra": Zero((), ((j, 2),), (), ScalarType.real),
        "extent": Zero((), ((i, 3),), (), ScalarType.real),
        "shape": Zero((2,), ((i, 2),), (), ScalarType.real),
        "arity": Zero((), ((i, 2),), (space("slot"),), ScalarType.real),
    }[kind]
    with pytest.raises(ValueError):
        complete_indices(original, replacement)
