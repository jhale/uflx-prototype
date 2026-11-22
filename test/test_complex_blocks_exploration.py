"""Mixed scalar differentiation and Jacobians using existing form slots."""

import basix
import numpy as np
import pytest
from test_basix import square
from testutil import push_forward

from uflx._src.extensions.complex_differentiation import complex_derivative_parts
from uflx.arithmetic import conjugate, imaginary_part, num, real_part
from uflx.calculus import directional_derivative, integral
from uflx.forms import form, function
from uflx.passes import apply_derivatives
from uflx_basix.assembly import conforming_dofmap
from uflx_basix.basix_cells import basix_cell
from uflx_basix.basix_spaces import basix_space
from uflx_basix.spaces import space
from uflx_compiler import assemble_bound
from uflx_oracle import evaluate

C, R = space("complex-block", scalar_type="complex"), space("real-block", scalar_type="real")
z, h, a = (function(C, name) for name in ("z", "h", "a"))
r, k, b = (function(R, name) for name in ("r", "k", "b"))


def value(f, data):
    return evaluate(apply_derivatives(f), data)(())


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
    with pytest.raises(ValueError, match="complex variable"):
        complex_derivative_parts(r * r, r, r, k)


@pytest.mark.parametrize("real_output", [False, True])
def test_assembled_mixed_jacobian_with_complex_and_real_direction_blocks(real_output):
    D = square(basix_cell(basix.CellType.triangle))[0]
    VC = basix_space("complex-state", D.topology, basix.ElementFamily.P, 1, scalar_type="complex")
    VR = basix_space("real-state", D.topology, basix.ElementFamily.P, 1, scalar_type="real")
    z, h = function(VC, "z"), function(VC, "h")
    r, k, test = (function(VR, name) for name in ("r", "k", "test"))
    zd, rd, vd = (push_forward(f, D) for f in (z, r, test))
    reaction = (
        real_part(zd * conjugate(zd)) - rd if real_output else zd * conjugate(zd) * zd + rd * zd
    )
    body = integral(reaction * vd, D)
    residual = form(body, (test,))
    plus, minus = complex_derivative_parts(body, z, z, h)
    blocks = (
        form(plus, (test, h)),
        form(minus, (test, conjugate(h))),
        form(directional_derivative(body, r, r, k), (test, k)),
    )
    Z = np.array([0.4 + 0.1j, -0.2 + 0.3j, 0.6 - 0.2j, 0.1 - 0.4j])
    H = np.array([0.2 - 0.1j, 0.4 + 0.2j, -0.1j, 0.3 + 0.1j])
    R = np.array([0.8, 0.5, -0.3, 0.6])
    K = np.array([-0.2, 0.4, 0.1, -0.1])
    bindings = {f: conforming_dofmap(f.space) for f in (z, h, r, k, test)}
    data = {"coefficients:z": Z, "coefficients:r": R}
    Jp, Jm, Jr = (assemble_bound(block, 6, bindings)(data) for block in blocks)
    residual_kernel = assemble_bound(residual, 6, bindings)
    step = 1e-6
    difference = (
        residual_kernel({"coefficients:z": Z + step * H, "coefficients:r": R + step * K})
        - residual_kernel({"coefficients:z": Z - step * H, "coefficients:r": R - step * K})
    ) / (2 * step)
    np.testing.assert_allclose(Jp @ H + Jm @ H.conj() + Jr @ K, difference, rtol=1e-7, atol=1e-9)

    # Real direction slots suffice, even for complex-valued output.
    dx = directional_derivative(body, z, z, k)
    dy = directional_derivative(body, z, z, num(1j) * k)
    Jx, Jy = (assemble_bound(form(d, (test, k)), 6, bindings)(data) for d in (dx, dy))
    np.testing.assert_allclose(Jx, Jp + Jm, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(Jy, 1j * (Jp - Jm), rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(Jx @ H.real + Jy @ H.imag + Jr @ K, difference, rtol=1e-7)

    # A completely real matrix uses projections before binding the real slots.
    for projection, extract in ((real_part, np.real), (imaginary_part, np.imag)):
        A = assemble_bound(form(projection(dx), (test, k)), 6, bindings)(data)
        B = assemble_bound(form(projection(dy), (test, k)), 6, bindings)(data)
        np.testing.assert_allclose(A, extract(Jx), atol=1e-12)
        np.testing.assert_allclose(B, extract(Jy), atol=1e-12)
