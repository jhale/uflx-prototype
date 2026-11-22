"""Contracts for form construction around one directional derivative primitive."""

import numpy as np
import pytest

from uflx._src.algorithms.substitution import substitute
from uflx._src.extensions.complex_differentiation import complex_derivative_parts
from uflx._src.nodes import DirectionalDerivative, Form
from uflx.arithmetic import conjugate, num
from uflx.calculus import directional_derivative
from uflx.forms import apply, form, function, partial
from uflx.nodes import ScalarType
from uflx.passes import apply_derivatives, expand_applications
from uflx_basix.spaces import space
from uflx_oracle import evaluate


def lower(e):
    return apply_derivatives(expand_applications(e))


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
    assert np.isclose(evaluate(lower(full), env)(()), expected)
    direct = directional_derivative(F, u, u, h)
    assert np.isclose(evaluate(lower(full), env)(()), evaluate(lower(direct), env)(()))
    if body is conjugate:
        assert np.isclose(evaluate(lower(apply(plus, (h,))), env)(()), 0)


def test_new_argument_follows_existing_form_arguments():
    V = space("real", scalar_type=ScalarType.real)
    u, v, w, h = (function(V, label) for label in ("u", "v", "w", "h"))
    F = form(u * u * v, (v,))
    e = form(directional_derivative(F, u, w), (w,))
    assert e.argument_spaces == (V, V)
    applied = lower(partial(e, {w: h}))
    assert applied.argument_spaces == (V,)
    assert np.isclose(evaluate(applied, {u: 2, h: 3})((5,)), 60)


def test_substitution_avoids_capturing_a_form_direction():
    V = space("capture", scalar_type=ScalarType.real)
    u, z, w = (function(V, label) for label in ("u", "z", "w"))
    e = form(directional_derivative(u * z, u, u, w), (w,))
    changed = substitute(e, {z: w})
    assert changed.arguments[0] is not w
    assert np.isclose(evaluate(lower(changed), {u: 2, w: 3})((7,)), 21)


def test_third_complex_derivative_has_the_expected_multilinear_value():
    V = space("higher")
    u = function(V, "u")
    e = u * u * u * u
    for n in range(3):
        e = form(directional_derivative(e, u, u, function(V, f"w{n}")), (function(V, f"w{n}"),))
    args = (0.2 + 0.1j, -0.3j, 0.7 - 0.1j)
    assert e.argument_spaces == (V, V, V)
    x = 0.7 + 0.2j
    assert np.isclose(evaluate(lower(e), {u: x})(args), 24 * x * np.prod(args))


def test_point_dependencies_remain_free_when_binding_a_fresh_direction():
    V = space("point", scalar_type=ScalarType.real)
    u, w, h = (function(V, name) for name in ("u", "w", "h"))
    e = form(directional_derivative(u * u, u, w + num(1), h), (h,))
    assert np.isclose(evaluate(lower(e), {w: 2})((3,)), 18)


@pytest.mark.parametrize("nonholomorphic", [False, True])
def test_complex_finite_element_jacobian_matches_a_residual_difference(nonholomorphic):
    import basix

    from uflx.calculus import gradient, integral
    from uflx_basix.assembly import conforming_dofmap
    from uflx_basix.basix_cells import basix_cell
    from uflx_basix.basix_spaces import basix_space
    from uflx_compiler import assemble_bound

    from ..testutil import push_forward
    from ..uflx_basix.test_basix import square

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

    from ..testutil import push_forward
    from ..uflx_basix.test_basix import square

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
