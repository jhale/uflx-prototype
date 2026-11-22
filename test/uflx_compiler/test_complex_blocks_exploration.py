"""Mixed scalar differentiation and Jacobians using existing form slots."""

import basix
import numpy as np
import pytest

from uflx._src.extensions.complex_differentiation import complex_derivative_parts
from uflx.arithmetic import conjugate, imaginary_part, num, real_part
from uflx.calculus import directional_derivative, integral
from uflx.forms import form, function
from uflx_basix.assembly import conforming_dofmap
from uflx_basix.basix_cells import basix_cell
from uflx_basix.basix_spaces import basix_space
from uflx_compiler import assemble_bound

from ..uflx_basix.meshes import push_forward, square


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
