"""General algebra and opaque library boundaries in derivative lowering."""

from __future__ import annotations

import numpy as np
import pytest

from uflx._src.extensions.complex_differentiation import complex_derivative_parts
from uflx._src.nodes import Complex, DirectionalDerivative
from uflx._src.passes import derivatives
from uflx.arithmetic import conjugate
from uflx.forms import form, function
from uflx.nodes import Integer
from uflx_basix.spaces import space
from uflx_oracle import evaluate


@pytest.mark.parametrize("antilinear", [False, True])
def test_rotated_direction_uses_the_general_differential(monkeypatch, antilinear):
    V = space("reuse")
    u, h = function(V, "u"), function(V, "h")
    body = conjugate(u * u) if antilinear else u * u
    calls = []
    original = derivatives.deriv

    def record(*args, **kwargs):
        calls.append(args)
        return original(*args, **kwargs)

    monkeypatch.setattr(derivatives, "deriv", record)
    out = derivatives.apply_derivatives(form(complex_derivative_parts(body, u, u, h)[0], (h,)))
    assert len(calls) == 2
    point, direction = 0.7 + 0.2j, 0.3 - 0.1j
    expected = 0 if antilinear else 2 * point * direction
    assert np.isclose(evaluate(out, {u: point})((direction,)), expected)


def test_direction_occurring_at_point_does_not_use_shortcut(monkeypatch):
    V = space("nonfresh-direction")
    u, h = function(V, "u"), function(V, "h")
    calls = []
    original = derivatives.deriv

    def record(*args, **kwargs):
        calls.append(args)
        return original(*args, **kwargs)

    monkeypatch.setattr(derivatives, "deriv", record)
    out = derivatives.apply_derivatives(
        DirectionalDerivative(u * u, u, h, Complex(Integer(0), Integer(1)) * h)
    )
    assert len(calls) == 1
    assert calls[0][3] is Complex(Integer(0), Integer(1)) * h
    point = 0.7 + 0.2j
    assert np.isclose(evaluate(out, {h: point})(()), 2j * point**2)


def test_mixed_differential_retains_general_projection(monkeypatch):
    V = space("mixed-fallback")
    u, h = function(V, "u"), function(V, "h")
    calls = []
    original = derivatives.deriv

    def record(*args, **kwargs):
        calls.append(args)
        return original(*args, **kwargs)

    monkeypatch.setattr(derivatives, "deriv", record)
    out = derivatives.apply_derivatives(
        form(complex_derivative_parts(u * conjugate(u), u, u, h)[0], (h,))
    )
    assert len(calls) == 2
    point, direction = 0.7 + 0.2j, 0.3 - 0.1j
    assert np.isclose(evaluate(out, {u: point})((direction,)), point.conjugate() * direction)
