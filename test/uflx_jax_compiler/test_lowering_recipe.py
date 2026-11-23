"""Completed recipes honor output contracts after substitution and extension work."""

import numpy as np

from uflx._src.algorithms.traversal import everywhere
from uflx._src.nodes import Apply, DirectionalDerivative
from uflx._src.passes.contracts import core_normal_form, derivatives_expanded
from uflx.arithmetic import conjugate, num
from uflx.calculus import directional_derivative
from uflx.forms import apply, dual_space, form, function
from uflx.passes import apply_derivatives, expand_applications, lower
from uflx_jax_compiler import trace

from ..uflx.spaces import space


def test_actions_exposed_by_binding_are_finished():
    V = space("nested-actions")
    u, v = function(V, "u"), function(V, "v")
    l = function(dual_space(V), "l")
    # Substituting a form for the functional exposes an action on that form.
    wrapper = form(apply(l, (conjugate(u),)), (l,))
    point = form(num(2) * conjugate(v), (conjugate(v),))
    result = expand_applications(apply(wrapper, (point,)))
    assert everywhere(result, lambda n: not isinstance(n, Apply))
    assert np.isclose(trace(result, u)(3 + 1j), 2 * (3 - 1j))


def test_derivative_finishes_actions_created_by_dual_direction():
    V = space("dual-direction")
    u, v = function(V, "u"), function(V, "v")
    l = function(dual_space(V), "l")
    direction = form(num(2) * conjugate(v), (conjugate(v),))
    e = directional_derivative(apply(l, (conjugate(u),)), l, l, direction)
    out = apply_derivatives(e)
    assert everywhere(out, lambda n: not isinstance(n, Apply | DirectionalDerivative))
    assert np.isclose(trace(out, u)(3 + 1j), 2 * (3 - 1j))
    assert not core_normal_form(lower(e))


def test_point_substitution_exposes_action_and_another_derivative():
    V = space("point-actions")
    u, v, h = (function(V, name) for name in ("u", "v", "h"))
    l = function(dual_space(V), "l")
    point = form(directional_derivative(conjugate(v) * u, u, u, h), (conjugate(v),))
    e = directional_derivative(apply(l, (conjugate(u),)) * u, l, point, point)
    out = apply_derivatives(e)
    assert not derivatives_expanded(out)
    assert everywhere(out, lambda n: not isinstance(n, Apply))
    assert np.isclose(trace(out, u, h)(2, 3), 12)


def test_lowering_a_derivative_in_a_geometry_direction():
    x, h = (function(space("shape-recipe"), name) for name in ("x", "h"))
    e = directional_derivative(x * x, x, x, h)
    out = lower(e)
    assert not core_normal_form(out)
    assert np.isclose(trace(out, x, h)(2, 3), 12)
