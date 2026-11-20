import numpy as np
import pytest
from testutil import ill_typed, keeps_static_type

from uflx_oracle import evaluate
from uflx_spec_interpreter import (
    ScalarType,
    absolute,
    conjugate,
    directional_derivative,
    form,
    function,
    multiply,
    num,
)
from uflx_spec_interpreter._src.extensions.complex_differentiation import complex_derivative_parts
from uflx_spec_interpreter._src.nodes import Conjugate
from uflx_spec_interpreter._src.nodes.forms import entry_function
from uflx_spec_interpreter._src.passes import applications, derivatives
from uflx_toy.spaces import space

# Derivatives in two unknowns at once, as for a mixed problem.
C, R = space("C"), space("R", scalar_type=ScalarType.real)
u, p, a0, a1, h0, h1 = (function(C, n) for n in ("u", "p", "a0", "a1", "h0", "h1"))
ENV = {u: 0.3 + 0.2j, p: -0.5 + 0.4j, a0: 0.7 - 0.1j, a1: 0.2 + 0.6j, h0: 0.4j, h1: 1.1}


def passes(e):
    out = derivatives.apply_derivatives(applications.expand_applications(e))
    assert not (ill_typed(e) or ill_typed(out)), ill_typed(e) or ill_typed(out)
    return out


def test_a_tuple_variable_moves_every_member_to_the_point():
    # At a point other than x, the differential of u p is h0 a1 + a0 h1, not h0 p + u h1.
    e = directional_derivative(multiply(u, p), (u, p), (a0, a1), (h0, h1))
    expected = ENV[h0] * ENV[a1] + ENV[a0] * ENV[h1]
    for f in (e, passes(e)):
        assert np.isclose(evaluate(f, ENV)(()), expected)
    assert not np.isclose(expected, ENV[h0] * ENV[p] + ENV[u] * ENV[h1])


def check(F, x, entry, values):
    # The passes keep the value of the explicitly projected derivative form, defined by the oracle.
    directions = tuple(entry_function(member) for member in entry)
    body = directional_derivative(F, x, x, directions)
    if x[0].space.scalar_type == "complex":
        body = complex_derivative_parts(F, x, x, directions)[int(isinstance(entry[0], Conjugate))]
    e = form(body, (entry,))
    out = passes(e)
    assert keeps_static_type(e, out)
    before, after = evaluate(e, ENV)((values,)), evaluate(out, ENV)((values,))
    assert np.isclose(before, after, rtol=1e-6), (before, after)
    return after


def test_a_real_mixed_derivative():
    x, y = function(R, "x"), function(R, "y")
    v, q = function(R, "v"), function(R, "q")
    ENV.update({x: 0.4, y: -0.3})
    # d(x^2 y) in (v, q) is 2 x y v + x^2 q.
    value = check(multiply(multiply(x, x), y), (x, y), (v, q), (0.5, 0.25))
    assert np.isclose(value, 2 * 0.4 * -0.3 * 0.5 + 0.16 * 0.25)


def test_a_holomorphic_mixed_derivative_and_its_zero_antilinear_part():
    v, q = function(C, "v"), function(C, "q")
    F = multiply(multiply(u, u), p)
    w = (0.3 - 0.2j, 0.1 + 0.5j)
    value = check(F, (u, p), (v, q), w)
    assert np.isclose(value, 2 * ENV[u] * ENV[p] * w[0] + ENV[u] ** 2 * w[1])
    assert np.isclose(check(F, (u, p), (conjugate(v), conjugate(q)), w), 0)


def test_a_mixed_derivative_that_is_not_holomorphic():
    v, q = function(C, "v"), function(C, "q")
    # |u|^2 p is not holomorphic in u, so both parts of the derivative are terms.
    F = multiply(absolute(multiply(u, u)), p)
    w = (0.2 + 0.1j, -0.4 + 0.3j)
    check(F, (u, p), (v, q), w)
    check(F, (u, p), (conjugate(v), conjugate(q)), w)


def test_a_constant_point_and_direction():
    e = directional_derivative(multiply(u, p), (u, p), (num(2), num(3)), (num(1), num(0)))
    assert np.isclose(evaluate(passes(e), ENV)(()), 3)


def test_mixed_scalar_variables_are_valid_but_a_mixed_product_slot_is_not():
    lam, v, mu = function(R, "lam"), function(C, "v"), function(R, "mu")
    d = directional_derivative(u * lam, (u, lam), (u, lam), (v, mu))
    data = {u: 0.3 + 0.2j, lam: 0.4, v: 0.5 - 0.1j, mu: -0.2}
    expected = data[v] * data[lam] + data[u] * data[mu]
    assert np.isclose(evaluate(passes(d), data)(()), expected)
    with pytest.raises(ValueError, match="one scalar type"):
        form(d, ((v, mu),))
