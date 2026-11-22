"""Contracts for form construction around one directional derivative primitive."""

import numpy as np
import pytest

from uflx._src.algorithms.substitution import substitute
from uflx._src.extensions.complex_differentiation import complex_derivative_parts
from uflx.arithmetic import conjugate, num
from uflx.calculus import directional_derivative
from uflx.forms import apply, form, function, partial
from uflx.nodes import ScalarType
from uflx.passes import apply_derivatives, expand_applications
from uflx_basix.spaces import space
from uflx_oracle.interpreter import Interpreter


def lower(e):
    return apply_derivatives(expand_applications(e))


@pytest.mark.parametrize("body", [conjugate, lambda u: u * conjugate(u)])
def test_both_complex_parts_reconstruct_the_real_differential(body):
    V = space("complex")
    u, w, h = (function(V, label) for label in ("u", "w", "h"))
    F = body(u)
    plus = form(complex_derivative_parts(F, u, u, w)[0], (w,))
    minus = form(complex_derivative_parts(F, u, u, w)[1], (conjugate(w),))
    full = apply(plus, (h,)) + apply(minus, (conjugate(h),))
    env = {u: 0.7 + 0.4j, h: 0.2 - 0.6j}
    expected = np.conj(env[h]) if body is conjugate else 2 * np.real(np.conj(env[u]) * env[h])
    oracle = Interpreter(env)
    assert np.isclose(oracle.evaluate(lower(full)), expected)
    direct = directional_derivative(F, u, u, h)
    assert np.isclose(oracle.evaluate(lower(full)), oracle.evaluate(lower(direct)))
    if body is conjugate:
        assert np.isclose(Interpreter(env).evaluate(lower(apply(plus, (h,)))), 0)


def test_new_argument_follows_existing_form_arguments():
    V = space("real", scalar_type=ScalarType.real)
    u, v, w, h = (function(V, label) for label in ("u", "v", "w", "h"))
    F = form(u * u * v, (v,))
    e = form(directional_derivative(F, u, w), (w,))
    assert e.argument_spaces == (V, V)
    applied = lower(partial(e, {w: h}))
    assert applied.argument_spaces == (V,)
    assert np.isclose(Interpreter({u: 2, h: 3}).evaluate(applied, arguments=(5,)), 60)


def test_substitution_avoids_capturing_a_form_direction():
    V = space("capture", scalar_type=ScalarType.real)
    u, z, w = (function(V, label) for label in ("u", "z", "w"))
    e = form(directional_derivative(u * z, u, u, w), (w,))
    changed = substitute(e, {z: w})
    assert changed.arguments[0] is not w
    assert np.isclose(Interpreter({u: 2, w: 3}).evaluate(lower(changed), arguments=(7,)), 21)


def test_third_complex_derivative_has_the_expected_multilinear_value():
    V = space("higher")
    u = function(V, "u")
    e = u * u * u * u
    for n in range(3):
        e = form(directional_derivative(e, u, u, function(V, f"w{n}")), (function(V, f"w{n}"),))
    args = (0.2 + 0.1j, -0.3j, 0.7 - 0.1j)
    assert e.argument_spaces == (V, V, V)
    x = 0.7 + 0.2j
    assert np.isclose(
        Interpreter({u: x}).evaluate(lower(e), arguments=args), 24 * x * np.prod(args)
    )


def test_point_dependencies_remain_free_when_binding_a_fresh_direction():
    V = space("point", scalar_type=ScalarType.real)
    u, w, h = (function(V, name) for name in ("u", "w", "h"))
    e = form(directional_derivative(u * u, u, w + num(1), h), (h,))
    assert np.isclose(Interpreter({w: 2}).evaluate(lower(e), arguments=(3,)), 18)
