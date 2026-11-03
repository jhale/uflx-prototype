"""Completed recipes honor output contracts after substitution and extension work."""

import numpy as np
import pytest

from uflx_oracle import evaluate
from uflx_spec_interpreter import (
    action,
    apply_actions,
    apply_derivatives,
    conjugate,
    directional_derivative,
    dual_space,
    form,
    function,
    lower,
    num,
)
from uflx_spec_interpreter._src.contracts import core_normal_form, derivatives_expanded
from uflx_spec_interpreter._src.nodes import Action, DirectionalDerivative
from uflx_spec_interpreter._src.substitution import everywhere
from uflx_toy.spaces import space


def test_actions_exposed_by_binding_are_finished():
    V = space("nested-actions")
    u, v = function(V, "u"), function(V, "v")
    l = function(dual_space(V), "l")
    # Substituting a form for the functional exposes an action on that form.
    wrapper = form(action(l, 0, conjugate(u)), (l,))
    point = form(num(2) * conjugate(v), (conjugate(v),))
    result = apply_actions(action(wrapper, 0, point))
    assert everywhere(result, lambda n: not isinstance(n, Action))
    assert np.isclose(evaluate(result, {u: 3 + 1j})(()), 2 * (3 - 1j))


def test_derivative_finishes_actions_created_by_dual_direction():
    V = space("dual-direction")
    u, v = function(V, "u"), function(V, "v")
    l = function(dual_space(V), "l")
    direction = form(num(2) * conjugate(v), (conjugate(v),))
    e = directional_derivative(action(l, 0, conjugate(u)), l, l, direction)
    out = apply_derivatives(e)
    assert everywhere(out, lambda n: not isinstance(n, Action | DirectionalDerivative))
    assert np.isclose(evaluate(out, {u: 3 + 1j})(()), 2 * (3 - 1j))
    assert not core_normal_form(lower(e))


def test_point_substitution_exposes_action_and_another_derivative():
    V = space("point-actions")
    u, v, h = (function(V, name) for name in ("u", "v", "h"))
    l = function(dual_space(V), "l")
    point = form(directional_derivative(conjugate(v) * u, u, u, h), (conjugate(v),))
    e = directional_derivative(action(l, 0, conjugate(u)) * u, l, point, point)
    out = apply_derivatives(e)
    assert not derivatives_expanded(out)
    assert everywhere(out, lambda n: not isinstance(n, Action))
    assert np.isclose(evaluate(out, {u: 2, h: 3})(()), 12)


def test_checked_lowering_rejects_extension_and_accepts_completed_extension_recipe():
    from uflx_spec_interpreter._draft import shape_derivative
    from uflx_spec_interpreter._src.extensions.shape import lower_with_shape_derivatives

    x, h = (function(space("shape-recipe"), name) for name in ("x", "h"))
    e = shape_derivative(x * x, x, h)
    with pytest.raises(TypeError, match="outside core normal form") as error:
        lower(e)
    assert error.value.args[1][0][0] is e
    out = lower_with_shape_derivatives(e)
    assert not core_normal_form(out)
    assert np.isclose(evaluate(out, {x: 2, h: 3})(()), 12)


def test_lowering_is_idempotent():
    x = function(space("lower-idempotent"), "x")
    out = lower(directional_derivative(x * x, x, x, num(1)))
    assert lower(out) is out
