"""Higher and mixed derivatives on fixed topology and coefficient data."""

import numpy as np
import pytest
from test_shape_derivatives import (
    COORDINATES,
    DIRECTION,
    RT,
    P,
    T,
    U,
    W,
    env,
    functionals,
    v0,
    v1,
    value,
    x0,
    x1,
)

from uflx.arithmetic import num
from uflx.calculus import directional_derivative
from uflx.forms import function
from uflx.passes import apply_derivatives
from uflx_basix.basix_spaces import basix_space
from uflx_compiler import compile_form, evaluate

XS, HS = (x0, x1), (v0, v1)


@pytest.mark.parametrize("name", tuple(functionals()))
def test_second_geometry_derivative_matches_direct_finite_difference(name):
    f = functionals()[name]
    first = apply_derivatives(directional_derivative(f, XS, XS, HS))
    second = apply_derivatives(directional_derivative(first, XS, XS, HS))
    nested = apply_derivatives(
        directional_derivative(directional_derivative(f, XS, XS, HS), XS, XS, HS)
    )
    assert nested is second
    step = 1e-4
    difference = (
        value(f, COORDINATES + step * DIRECTION)
        - 2 * value(f, COORDINATES)
        + value(f, COORDINATES - step * DIRECTION)
    ) / step**2
    assert np.isclose(value(second, COORDINATES), difference, rtol=2e-6, atol=1e-6)


@pytest.mark.parametrize(
    "name,label,family,degree,coefficients",
    [("energy", "u", P, 2, U), ("mass of a Raviart-Thomas function", "w", RT, 1, W)],
)
def test_mixed_geometry_field_derivatives_in_both_orders(name, label, family, degree, coefficients):
    f = functionals()[name]
    V = basix_space("V" if label == "u" else "R", T, family, degree)
    u, h = function(V, label), function(V, "perturbation")
    perturbation = np.random.default_rng(81).normal(size=coefficients.shape)
    dxdu = apply_derivatives(directional_derivative(directional_derivative(f, u, u, h), XS, XS, HS))
    dudx = apply_derivatives(directional_derivative(directional_derivative(f, XS, XS, HS), u, u, h))
    data = env(COORDINATES) | {"coefficients:perturbation": perturbation}
    a, b = (evaluate(compile_form(g, 8), data) for g in (dxdu, dudx))
    assert np.isclose(a, b, rtol=1e-10, atol=1e-10)
    kernel = compile_form(f, 8)
    step = 1e-4

    def shifted(sx, su):
        data = env(COORDINATES + sx * step * DIRECTION)
        data[f"coefficients:{label}"] = coefficients + su * step * perturbation
        return evaluate(kernel, data)

    finite_difference = (shifted(1, 1) - shifted(1, -1) - shifted(-1, 1) + shifted(-1, -1)) / (
        4 * step**2
    )
    assert abs(a) > 1e-3  # A nontrivial direction, not a constant-field nullspace.
    assert np.isclose(a, finite_difference, rtol=2e-6, atol=1e-6)


def test_geometry_dependent_direction_participates_in_outer_derivative():
    f = functionals()["energy"]
    inner = directional_derivative(f, XS, XS, XS)
    result = apply_derivatives(directional_derivative(inner, XS, XS, HS))
    kernel = compile_form(apply_derivatives(directional_derivative(f, XS, XS, XS)), 8)
    step = 1e-6
    difference = (
        evaluate(kernel, env(COORDINATES + step * DIRECTION))
        - evaluate(kernel, env(COORDINATES - step * DIRECTION))
    ) / (2 * step)
    assert np.isclose(value(result, COORDINATES), difference, rtol=1e-6, atol=1e-7)


def test_geometry_dependent_evaluation_point_participates_in_outer_derivative():
    f = functionals()["area"]
    point = tuple(num(2) * x for x in XS)
    inner = directional_derivative(f, XS, point, HS)
    result = apply_derivatives(directional_derivative(inner, XS, XS, HS))
    kernel = compile_form(apply_derivatives(directional_derivative(f, XS, point, HS)), 8)
    step = 1e-6
    difference = (
        evaluate(kernel, env(COORDINATES + step * DIRECTION))
        - evaluate(kernel, env(COORDINATES - step * DIRECTION))
    ) / (2 * step)
    assert abs(difference) > 1e-3
    assert np.isclose(value(result, COORDINATES), difference, rtol=1e-6, atol=1e-7)
