"""General algebra and opaque library boundaries in derivative lowering."""

from __future__ import annotations

from functools import cached_property

import numpy as np
import pytest

from uflx_oracle import evaluate
from uflx_spec_interpreter import (
    action,
    apply_actions,
    apply_derivatives,
    conjugate,
    dual_space,
    function,
    gateaux_derivative,
    index,
    num,
    tensor,
)
from uflx_spec_interpreter._src.nodes import AbstractExpression, Action, Product, Zero
from uflx_spec_interpreter._src.passes import derivatives
from uflx_spec_interpreter._src.passes.algebra import normalizer
from uflx_spec_interpreter._src.substitution import bottom_up, operands
from uflx_toy.spaces import space


def nodes(e):
    seen = set()

    def walk(e):
        if e not in seen:
            seen.add(e)
            for child in operands(e):
                walk(child)

    walk(e)
    return seen


def lower(e):
    return apply_derivatives(apply_actions(e))


def test_collection_keeps_cancelled_free_indices():
    k = index("k")
    indexed = tensor((num(2), num(3)))[k]
    out = normalizer()(indexed - indexed)
    assert isinstance(out, Zero)
    assert out.free_indices == ((k, 2),)


def test_scalar_collection_is_idempotent_without_product_expansion():
    V = space("algebra")
    a, b, c, d = (function(V, label) for label in "abcd")
    e = (a + b) * (c + d)
    out = normalizer()(e)
    assert normalizer()(out) is out
    assert isinstance(out, Product)


def test_opaque_library_dual_function_needs_no_custom_simplification():
    V = space("opaque-linear")
    u, w = function(V, "u"), function(V, "w")
    L = function(dual_space(V), "library-operator")
    F = action(L, 0, u * u)
    plus = lower(gateaux_derivative(F, u, u, w))
    minus = lower(gateaux_derivative(F, u, u, conjugate(w)))
    assert not any(isinstance(n, Action) for n in nodes(minus))
    env = {u: 0.7 + 0.2j, L: lambda z: (2 + 3j) * z}
    h = 0.3 - 0.1j
    assert np.isclose(evaluate(plus, env)((h,)), (2 + 3j) * 2 * env[u] * h)
    assert evaluate(minus, env)((h,)) == 0


class LibraryValue(AbstractExpression):
    """An opaque scalar library evaluation; the core cannot inspect its algorithm."""

    identifier = "experiment.LibraryValue"
    operand: AbstractExpression
    value_shape = ()

    @property
    def free_indices(self):
        return self.operand.free_indices

    @property
    def argument_spaces(self):
        return self.operand.argument_spaces

    @property
    def support(self):
        return self.operand.support

    @property
    def scalar_type(self):
        return self.operand.scalar_type


class LibrarySlope(LibraryValue):
    """A coefficient of a library differential, also evaluated outside the core."""

    identifier = "experiment.LibrarySlope"
    operand: AbstractExpression


class LibraryJVP(AbstractExpression):
    """A callback whose dependence on the direction is not exposed."""

    identifier = "experiment.LibraryJVP"
    point: AbstractExpression
    direction: AbstractExpression
    value_shape = ()

    @cached_property
    def free_indices(self):
        return tuple(dict(self.point.free_indices + self.direction.free_indices).items())

    @property
    def argument_spaces(self):
        return self.direction.argument_spaces

    @property
    def support(self):
        return self.point.support or self.direction.support

    @property
    def scalar_type(self):
        return self.joined_scalar_type(self.point, self.direction)


@pytest.mark.parametrize("expose_direction", [False, True])
def test_library_rule_can_expose_direction_without_exposing_its_algorithm(
    monkeypatch, expose_direction
):
    original = derivatives.derivative_of_node

    def library_rule(e, d, constant, avoid):
        if type(e) is LibraryValue:
            if expose_direction:
                return Product(LibrarySlope(e.operand), d(e.operand))
            return LibraryJVP(e.operand, d(e.operand))
        return original(e, d, constant, avoid)

    # Test adapter for an extension-supplied derivative rule, not a new public registry.
    monkeypatch.setattr(derivatives, "derivative_of_node", library_rule)
    V = space("opaque-nonlinear")
    u, w = function(V, "u"), function(V, "w")
    a, h = num(0.7 + 0.2j), num(0.3 - 0.1j)
    plus = gateaux_derivative(LibraryValue(u), u, a, w)
    minus = gateaux_derivative(LibraryValue(u), u, a, conjugate(w))
    symbolic_minus = lower(minus)
    if expose_direction:
        assert not any(isinstance(n, LibrarySlope | LibraryJVP) for n in nodes(symbolic_minus))
    else:
        assert sum(isinstance(n, LibraryJVP) for n in nodes(symbolic_minus)) == 2

    def supply_library_values(e):
        if isinstance(e, LibrarySlope):
            return num(3 * evaluate(e.operand, {})(()) ** 2)
        if isinstance(e, LibraryJVP):
            return num(3 * evaluate(e.point, {})(()) ** 2 * evaluate(e.direction, {})(()))
        return e

    # Use x^3 as the externally implemented example; the core never sees that body.
    for derivative, argument, expected in (
        (plus, h, 3 * (0.7 + 0.2j) ** 2 * (0.3 - 0.1j)),
        (minus, conjugate(h), 0),
    ):
        out = lower(action(derivative, 0, argument))
        resolved = bottom_up(out, supply_library_values)
        assert np.isclose(evaluate(resolved, {})(()), expected)


@pytest.mark.parametrize("shape", [(), (2,)])
def test_zero_action_keeps_family_indices_and_argument_shape(shape):
    V = space("zero-action", value_shape=shape)
    L = function(dual_space(V), "L")
    k = index("family")
    out = normalizer()(action(L, 0, Zero(shape, ((k, 3),), ())))
    assert isinstance(out, Zero)
    assert out.value_shape == ()
    assert out.free_indices == ((k, 3),)
    assert out.argument_spaces == ()


def test_complex_coefficients_have_a_stable_collected_representation():
    from uflx_spec_interpreter._src.nodes import ImaginaryUnit

    u = function(space("complex-coefficients"), "u")
    i = ImaginaryUnit()
    first = normalizer()(u + i * u)
    second = normalizer()((num(1) + i) * u)
    assert first is second
    assert normalizer()(first) is first
    assert isinstance(normalizer()(first - second), Zero)
