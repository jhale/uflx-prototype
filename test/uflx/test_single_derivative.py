"""Contracts for form construction around one directional derivative primitive."""

import pytest

from uflx._src.algorithms.substitution import substitute
from uflx._src.nodes import DirectionalDerivative, Form
from uflx.arithmetic import conjugate
from uflx.calculus import directional_derivative
from uflx.forms import form, function
from uflx.nodes import ScalarType

from .spaces import space


def test_derivative_at_the_variable_is_the_same_node():
    V = space("shorthand", scalar_type=ScalarType.real)
    u, v, a, h, k = (function(V, name) for name in "uvahk")
    body = u * v
    for variable, point, direction in ((u, a, h), ((u, v), (a, v), (h, k))):
        expected = DirectionalDerivative(body, variable, variable, direction)
        assert directional_derivative(body, variable, direction) is expected
        assert (
            directional_derivative(expression=body, variable=variable, direction=direction)
            is expected
        )
        assert directional_derivative(body, variable, variable, direction) is expected
        at_point = DirectionalDerivative(body, variable, point, direction)
        assert directional_derivative(body, variable, point, direction=direction) is at_point
        assert directional_derivative(body, variable, point=point, direction=direction) is at_point
        assert substitute(expected, {u: a}) is at_point
    with pytest.raises(TypeError):
        directional_derivative(body, u, point=a)


@pytest.mark.parametrize("complex_", [False, True])
def test_factory_is_interned_and_point_substitution_commutes(complex_):
    V = space("factory", scalar_type=ScalarType.complex if complex_ else ScalarType.real)
    u, a, b, w = (function(V, label) for label in ("u", "a", "b", "w"))
    F = u * u
    e = form(directional_derivative(F, u, a, w), (w,))
    assert form(directional_derivative(F, u, a, w), (w,)) is e
    assert substitute(e, {a: b}) is form(directional_derivative(F, u, b, w), (w,))
    if not complex_:
        assert e is Form(DirectionalDerivative(F, u, a, w), (w,))


@pytest.mark.parametrize("conjugated", [False, True])
def test_duplicate_tuple_direction_binders_are_rejected(conjugated):
    V = space("tuple")
    u, v, w = (function(V, label) for label in ("u", "v", "w"))
    entry = conjugate(w) if conjugated else w
    with pytest.raises(ValueError, match="form binders must be distinct"):
        form(directional_derivative(u * v, (u, v), (u, v), (entry, entry)), ((entry, entry),))
