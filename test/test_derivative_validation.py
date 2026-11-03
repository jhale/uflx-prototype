"""Static admissibility of the derivative-form construction function."""

import pytest

from uflx_spec_interpreter import (
    ScalarType,
    action,
    conjugate,
    dual_space,
    form,
    function,
    gateaux_derivative,
    index,
    num,
    tensor,
)
from uflx_spec_interpreter._src.nodes import RD, ConjugateSpace
from uflx_toy.spaces import space


@pytest.fixture
def variables():
    V = space("validation")
    return tuple(function(V, label) for label in ("u", "v", "w", "z"))


@pytest.mark.parametrize(
    ("arguments", "error", "message"),
    [
        (lambda u, v, w, z: (u, (), (), ()), ValueError, "at least two"),
        (lambda u, v, w, z: (u, (u,), (u,), (w,)), ValueError, "at least two"),
        (lambda u, v, w, z: (u, u, (u,), w), ValueError, "tuple structure"),
        (lambda u, v, w, z: (u, (u, v), (u,), (w, z)), ValueError, "lengths"),
        (lambda u, v, w, z: (u, (u, u), (u, v), (w, z)), ValueError, "distinct"),
        (lambda u, v, w, z: (u, u, u, conjugate(u * w)), ValueError, "form arguments"),
        (lambda u, v, w, z: (u, u, tensor((num(1), num(2))), w), ValueError, "admissible"),
        (lambda u, v, w, z: (u, u, form(v, (v,)), w), ValueError, "admissible"),
        (lambda u, v, w, z: (u * w, u, u, w), ValueError, "must not be free"),
        (
            lambda u, v, w, z: (u, (u, v), (u, v), (w, conjugate(z))),
            ValueError,
            "all functions or all conjugates",
        ),
    ],
)
def test_rejected_inputs(variables, arguments, error, message):
    with pytest.raises(error, match=message):
        gateaux_derivative(*arguments(*variables))


def test_real_variable_rejects_complex_point_and_mixed_tuple(variables):
    u, _, w, _ = variables
    R = space("real-validation", scalar_type=ScalarType.real)
    r, s = function(R, "r"), function(R, "s")
    with pytest.raises(ValueError, match="admissible"):
        gateaux_derivative(r, r, num(1j), s)
    with pytest.raises(ValueError, match="mixed real and complex"):
        gateaux_derivative(r * u, (r, u), (r, u), (s, w))
    with pytest.raises(ValueError, match="variable spaces"):
        gateaux_derivative(u, u, u, s)


def test_point_support_and_no_support():
    V = space("supported", domain=RD(2))
    u, w = function(V, "u"), function(V, "w")
    other = function(space("other-domain", domain=RD(3)), "a")
    with pytest.raises(ValueError, match="admissible"):
        gateaux_derivative(u, u, other, w)
    assert gateaux_derivative(u, u, num(2), w).argument_spaces == (V,)
    assert gateaux_derivative(u, u, function(V, "a"), w).argument_spaces == (V,)


def test_nonbindable_function_is_rejected(variables):
    u, _, _, _ = variables
    operator_space = ConjugateSpace(dual_space(u.space))
    x, h = function(operator_space, "x"), function(operator_space, "h")
    with pytest.raises(ValueError, match="bindable"):
        gateaux_derivative(x, x, x, h)


def test_dual_variable_accepts_symbolic_form_point(variables):
    u, v, _, _ = variables
    D = dual_space(u.space)
    x, h = function(D, "load"), function(D, "direction")
    point = form(conjugate(v), (conjugate(v),))
    F = action(x, 0, conjugate(u))
    assert gateaux_derivative(F, x, point, h).argument_spaces == (D,)
    wrong = form(v, (v,))
    with pytest.raises(ValueError, match="admissible"):
        gateaux_derivative(F, x, wrong, h)


def test_family_extents_are_checked_between_body_and_all_points(variables):
    u, v, w, z = variables
    k = index("family-validation")
    a = tensor((num(1), num(2)))[k]
    b = tensor((num(1), num(2), num(3)))[k]
    with pytest.raises(ValueError, match="matching extents"):
        gateaux_derivative(u * a, u, b, w)
    with pytest.raises(ValueError, match="matching extents"):
        gateaux_derivative(u * v, (u, v), (a, b), (w, z))
    out = gateaux_derivative(u * a, u, a, w)
    assert out.free_indices == ((k, 2),)


def test_bound_direction_is_not_a_free_collision_and_real_points_are_allowed(variables):
    u, v, w, z = variables
    F = form(u * w, (w,))
    out = gateaux_derivative(F, u, num(2), w)
    assert out.argument_spaces == (u.space, u.space)
    joint = gateaux_derivative(u * v, (u, v), (num(2), num(3)), (w, z))
    assert joint.argument_spaces == ((u.space, v.space),)
