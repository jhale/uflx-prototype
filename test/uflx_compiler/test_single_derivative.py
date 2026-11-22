"""Contracts for form construction around one directional derivative primitive."""

import numpy as np
import pytest

from uflx._src.extensions.complex_differentiation import complex_derivative_parts
from uflx.arithmetic import conjugate, num
from uflx.forms import apply, form, function
from uflx.nodes import ScalarType


@pytest.mark.parametrize("nonholomorphic", [False, True])
def test_complex_finite_element_jacobian_matches_a_residual_difference(nonholomorphic):
    import basix

    from uflx.calculus import gradient, integral
    from uflx_basix.assembly import conforming_dofmap
    from uflx_basix.basix_cells import basix_cell
    from uflx_basix.basix_spaces import basix_space
    from uflx_compiler import assemble_bound

    from ..uflx_basix.meshes import push_forward, square

    D = square(basix_cell(basix.CellType.triangle))[0]
    V = basix_space(
        "complex-fe", D.topology, basix.ElementFamily.P, 1, scalar_type=ScalarType.complex
    )
    uT, vT, wT = (function(V, label) for label in ("u", "v", "w"))
    u, v = push_forward(uT, D), push_forward(vT, D)
    gu, gv = gradient(u, D), gradient(v, D)
    reaction = u * conjugate(u) * u if nonholomorphic else u
    residual = form(
        integral(
            gu[0] * conjugate(gv[0]) + gu[1] * conjugate(gv[1]) - num(2) * reaction * conjugate(v),
            D,
        ),
        (conjugate(vT),),
    )
    plus = form(complex_derivative_parts(residual, uT, uT, wT)[0], (wT,))
    minus = form(complex_derivative_parts(residual, uT, uT, wT)[1], (conjugate(wT),))
    U = np.array([0.4 + 0.1j, -0.2 + 0.3j, 0.6 - 0.2j, 0.1 - 0.4j])
    H = np.array([0.2 - 0.1j, 0.4 + 0.2j, -0.1j, 0.3 + 0.1j])
    env = {"coefficients:u": U}
    mapping = conforming_dofmap(V)
    bindings = {uT: mapping, vT: mapping, wT: mapping}
    Jp, Jm = assemble_bound(plus, 5, bindings)(env), assemble_bound(minus, 5, bindings)(env)
    R = assemble_bound(residual, 5, bindings)
    step = 1e-6
    difference = (R({"coefficients:u": U + step * H}) - R({"coefficients:u": U - step * H})) / (
        2 * step
    )
    np.testing.assert_allclose(Jp @ H + Jm @ H.conj(), difference, rtol=1e-7, atol=1e-9)
    if not nonholomorphic:
        np.testing.assert_allclose(Jm, 0, atol=1e-12)


@pytest.mark.parametrize("nonholomorphic", [False, True])
def test_applied_complex_fe_differential_matches_finite_difference(nonholomorphic):
    import basix

    from uflx.calculus import gradient, integral
    from uflx_basix.assembly import conforming_dofmap
    from uflx_basix.basix_cells import basix_cell
    from uflx_basix.basix_spaces import basix_space
    from uflx_compiler import compile_form
    from uflx_compiler import evaluate as compiled_value

    from ..uflx_basix.meshes import push_forward, square

    D = square(basix_cell(basix.CellType.triangle))[0]
    V = basix_space(
        "complex-action", D.topology, basix.ElementFamily.P, 1, scalar_type=ScalarType.complex
    )
    uT, vT, wT, hT, zT = (function(V, label) for label in ("u", "v", "w", "h", "z"))
    u, v = push_forward(uT, D), push_forward(vT, D)
    gu, gv = gradient(u, D), gradient(v, D)
    reaction = u * conjugate(u) * u if nonholomorphic else u
    residual = form(
        integral(
            gu[0] * conjugate(gv[0]) + gu[1] * conjugate(gv[1]) - num(2) * reaction * conjugate(v),
            D,
        ),
        (conjugate(vT),),
    )
    plus = form(complex_derivative_parts(residual, uT, uT, wT)[0], (wT,))
    minus = form(complex_derivative_parts(residual, uT, uT, wT)[1], (conjugate(wT),))
    applied = apply(plus, (conjugate(zT), hT)) + apply(minus, (conjugate(zT), conjugate(hT)))
    derivative = compile_form(applied, 5)
    primal = compile_form(apply(residual, (conjugate(zT),)), 5)
    U = np.array([0.4 + 0.1j, -0.2 + 0.3j, 0.6 - 0.2j, 0.1 - 0.4j])
    H = np.array([0.2 - 0.1j, 0.4 + 0.2j, -0.1j, 0.3 + 0.1j])
    Z = np.array([0.1j, 0.2 - 0.3j, 0.7 + 0.1j, 0.4])
    mapping = conforming_dofmap(V)
    U, H, Z = (mapping.gather(values) for values in (U, H, Z))
    env = {"coefficients:u": U, "coefficients:h": H, "coefficients:z": Z}
    step = 1e-6
    difference = (
        compiled_value(primal, env | {"coefficients:u": U + step * H})
        - compiled_value(primal, env | {"coefficients:u": U - step * H})
    ) / (2 * step)
    np.testing.assert_allclose(compiled_value(derivative, env), difference, rtol=1e-7, atol=1e-9)
