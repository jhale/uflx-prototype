"""Admissibility belongs to the derivative; slot declarations belong to Form."""

import pytest

from uflx_spec_interpreter import (
    EuclideanSpace,
    apply,
    apply_derivatives,
    conjugate,
    directional_derivative,
    dual_space,
    form,
    function,
    index,
    num,
    tensor,
)
from uflx_spec_interpreter._src.nodes import ConjugateSpace
from uflx_toy.spaces import space


@pytest.fixture
def variables():
    V = space("validation")
    return tuple(function(V, label) for label in ("u", "v", "w", "z"))


@pytest.mark.parametrize(
    "arguments,message",
    [
        (lambda u, v, w, z: (u, (), (), ()), "at least two"),
        (lambda u, v, w, z: (u, (u,), (u,), (w,)), "at least two"),
        (lambda u, v, w, z: (u, u, (u,), w), "tuple structure"),
        (lambda u, v, w, z: (u, (u, v), (u,), (w, z)), "tuple structure"),
        (lambda u, v, w, z: (u, (u, u), (u, v), (w, z)), "distinct"),
        (lambda u, v, w, z: (u, u, tensor((num(1), num(2))), w), "value shape"),
        (lambda u, v, w, z: (u, u, form(v, (v,)), w), "argument spaces"),
    ],
)
def test_rejected_derivative_inputs(variables, arguments, message):
    with pytest.raises(ValueError, match=message):
        directional_derivative(*arguments(*variables))


def test_real_variable_rejects_complex_point_but_mixed_variables_are_valid(variables):
    u, _, w, _ = variables
    R = space("real-validation", scalar_type="real")
    r, s = function(R, "r"), function(R, "s")
    with pytest.raises(ValueError, match="scalar type"):
        directional_derivative(r, r, num(1j), s)
    d = directional_derivative(r * u, (r, u), (r, u), (s, w))
    assert d.scalar_type == "complex"
    assert d.argument_spaces == ()
    # A real direction in a compatible space is admissible for a complex variable.
    assert form(directional_derivative(u, u, u, s), (s,)).argument_spaces == (R,)
    with pytest.raises(ValueError, match="one scalar type"):
        form(d, ((s, w),))


def test_directions_are_expressions_not_declarations(variables):
    u, v, w, z = variables
    d = directional_derivative(u * v, (u, v), (u, v), (w, conjugate(z)))
    assert d.argument_spaces == ()
    assert directional_derivative(u, u, w + num(1), w).point is w + num(1)
    # Repeated directions are valid; repeated form declarations are not.
    d = directional_derivative(u * v, (u, v), (u, v), (w, w))
    with pytest.raises(ValueError, match="form binders must be distinct"):
        form(d, ((w, w),))
    with pytest.raises(ValueError, match="form arguments"):
        form(d, (conjugate(u * w),))


def test_point_support_and_no_support():
    V = space("supported", domain=EuclideanSpace(2))
    u, w = function(V, "u"), function(V, "w")
    other = function(space("other-domain", domain=EuclideanSpace(3)), "a")
    with pytest.raises(ValueError, match="domain"):
        directional_derivative(u, u, other, w)
    for point in (num(2), function(V, "a")):
        assert form(directional_derivative(u, u, point, w), (w,)).argument_spaces == (V,)


def test_map_valued_variable_is_bindable(variables):
    u, _, _, _ = variables
    operator_space = ConjugateSpace(dual_space(u.space))
    x, h = function(operator_space, "x"), function(operator_space, "h")
    assert apply_derivatives(directional_derivative(x, x, x, h)) is h
    assert form(x, (x,)).argument_spaces == (*x.argument_spaces, operator_space)


def test_dual_variable_accepts_symbolic_form_point(variables):
    u, v, _, _ = variables
    D = dual_space(u.space)
    x, h = function(D, "load"), function(D, "direction")
    point = form(conjugate(v), (conjugate(v),))
    F = apply(x, (conjugate(u),))
    assert form(directional_derivative(F, x, point, h), (h,)).argument_spaces == (D,)
    with pytest.raises(ValueError, match="argument spaces"):
        directional_derivative(F, x, form(v, (v,)), h)


def test_family_extents_are_checked_between_body_points_and_directions(variables):
    u, v, w, z = variables
    k = index("family-validation")
    a = tensor((num(1), num(2)))[k]
    b = tensor((num(1), num(2), num(3)))[k]
    with pytest.raises(ValueError, match="inconsistent extents"):
        directional_derivative(u * a, u, b, w)
    with pytest.raises(ValueError, match="inconsistent extents"):
        directional_derivative(u * v, (u, v), (a, b), (w, z))
    with pytest.raises(ValueError, match="inconsistent extents"):
        directional_derivative(u, u, a, b)
    assert directional_derivative(u * a, u, a, w).free_indices == ((k, 2),)


def test_explicit_binding_adds_slots_after_existing_slots(variables):
    u, v, w, z = variables
    F = form(u * w, (w,))
    out = form(directional_derivative(F, u, num(2), z), (z,))
    assert out.argument_spaces == (u.space, u.space)
    # u*v is holomorphic, so its full differential can be bound complex-linearly.
    joint = form(directional_derivative(u * v, (u, v), (num(2), num(3)), (w, z)), ((w, z),))
    assert joint.argument_spaces == ((u.space, v.space),)


def test_only_one_derivative_primitive_is_exported():
    import uflx_spec_interpreter as api

    assert "directional_derivative" in api.__all__
    assert "gateaux_derivative" not in api.__all__
    assert not hasattr(api, "gateaux_derivative")


def test_different_slots_may_have_different_scalar_types(variables):
    u, _, _, _ = variables
    r = function(space("real-slot", scalar_type="real"), "r")
    f = form(u * r, (u, r))
    assert f.argument_spaces == (u.space, r.space)
