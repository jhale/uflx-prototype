"""Mixed scalar differentiation and Jacobians using existing form slots."""

import numpy as np
import pytest

from uflx.arithmetic import conjugate, num, real_part
from uflx.calculus import directional_derivative
from uflx.forms import function
from uflx.passes import apply_derivatives
from uflx_extensions.complex_differentiation import complex_derivative_parts
from uflx_jax_compiler import trace

from ..uflx.spaces import space

C, R = space("complex-block", scalar_type="complex"), space("real-block", scalar_type="real")
z, h, a = (function(C, name) for name in ("z", "h", "a"))
r, k, b = (function(R, name) for name in ("r", "k", "b"))


def value(f, data):
    return trace(apply_derivatives(f), *data)(*data.values())


@pytest.mark.parametrize("real_output", [False, True])
def test_mixed_differential_and_explicit_parts_match_finite_difference(real_output):
    f = real_part(z * conjugate(z)) + r * r if real_output else z * conjugate(z) * z + r * z
    data = {a: 0.7 + 0.4j, h: 0.2 - 0.6j, b: 0.8, k: -0.3}
    full = directional_derivative(f, (z, r), (a, b), (h, k))
    # Differentiate each independent block, then evaluate the other block at its point.
    from uflx.algorithms import replace

    plus, minus = complex_derivative_parts(replace(f, {r: b}), z, a, h)
    real_block = directional_derivative(replace(f, {z: a}), r, b, k)
    assert np.isclose(value(full, data), value(plus + minus + real_block, data))
    step = 1e-6
    plus_data = {z: data[a] + step * data[h], r: data[b] + step * data[k]}
    minus_data = {z: data[a] - step * data[h], r: data[b] - step * data[k]}
    difference = (value(f, plus_data) - value(f, minus_data)) / (2 * step)
    assert np.isclose(value(full, data), difference, rtol=1e-8)
    assert full.scalar_type == ("real" if real_output else "complex")
    if real_output:
        assert plus.scalar_type == minus.scalar_type == "complex"
        assert abs(value(plus, data).imag) > 1e-3
        assert np.isclose(value(plus, data), np.conj(value(minus, data)))


def test_full_differential_is_real_linear_but_not_complex_linear():
    f = z * conjugate(z)
    data = {z: 0.7 + 0.4j, h: 0.2 - 0.6j}
    d = directional_derivative(f, z, z, h)
    di = directional_derivative(f, z, z, num(1j) * h)
    assert not np.isclose(value(di, data), 1j * value(d, data))
    assert np.isclose(value(d, data | {h: 2 * data[h]}), 2 * value(d, data))


@pytest.mark.parametrize("body", [lambda z: z * z, conjugate])
def test_projection_keeps_complex_linearity_explicit(body):
    plus, minus = complex_derivative_parts(body(z), z, z, h)
    data = {z: 0.7 + 0.4j, h: 0.2 - 0.6j}
    rotated = data | {h: 1j * data[h]}
    assert np.isclose(value(plus, rotated), 1j * value(plus, data))
    assert np.isclose(value(minus, rotated), -1j * value(minus, data))
