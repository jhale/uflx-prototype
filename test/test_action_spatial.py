"""Spatial operators transform a form's output without transforming its arguments."""

from __future__ import annotations

import numpy as np
import pytest

from uflx_oracle import at, evaluate
from uflx_spec_interpreter import (
    RD,
    ScalarType,
    action,
    apply_actions,
    apply_gradients,
    coordinate,
    domain_map,
    expand_pull_backs,
    form,
    function,
    gradient,
    index,
    lower,
    num,
    pull_back,
    tensor,
)
from uflx_spec_interpreter._src.nodes import AbstractDomain, AbstractFunctionSpace, Action, SideMap
from uflx_spec_interpreter._src.passes.actions import _actions_complete
from uflx_spec_interpreter._src.passes.contracts import _core_normal_form
from uflx_spec_interpreter._src.passes.lowering import _lower_spatial as lower_spatial
from uflx_spec_interpreter._src.passes.pullbacks import _pull_backs_expanded
from uflx_toy.cells import reference_simplex
from uflx_toy.spaces import space


class OperatorSpace(AbstractFunctionSpace):
    """A library-supplied operator with field-valued output and specified slots."""

    identifier = "test.OperatorSpace"
    domain: AbstractDomain
    argument_spaces: tuple
    value_shape = ()
    scalar_type = ScalarType.real


def setup():
    P, K = RD(1), reference_simplex(1)
    x, t = coordinate(P)[0], coordinate(K)[0]
    v = function(space("spatial-argument", domain=P, scalar_type=ScalarType.real), "v")
    A = form(x * v, (v,))
    psi = domain_map(K, tensor((num(1) + num(2) * t,)), P)
    return P, K, x, v, A, psi


def value(e, domain, point):
    return np.asarray(at(domain, np.array([point]), lambda: evaluate(e, {})(()))).real


def test_pullback_keeps_the_original_field_argument_and_value():
    P, K, x, _, A, psi = setup()
    g = x * x
    e = pull_back(action(gradient(A, P), 0, g), psi)
    out = expand_pull_backs(e)
    assert isinstance(out, Action)
    assert out.argument is g and out.argument.domain is P
    assert out.argument_spaces == e.argument_spaces
    np.testing.assert_allclose(value(out, K, 0.3), [3 * 1.6**2], atol=1e-6)
    np.testing.assert_allclose(value(out, K, 0.3), value(e, K, 0.3), atol=1e-6)
    complete = lower(e)
    assert not _core_normal_form(complete)
    assert lower_spatial(complete) is complete
    np.testing.assert_allclose(value(complete, K, 0.3), [3 * 1.6**2], atol=1e-5)


@pytest.mark.parametrize("retained", [False, True])
def test_gradient_does_not_count_argument_variation_twice(retained):
    P, _, x, _, A, _ = setup()
    f = gradient(A, P) if retained else A
    e = gradient(action(f, 0, x * x), P)
    out = apply_gradients(e)
    expected = [[6 * 1.6]] if retained else [3 * 1.6**2]
    np.testing.assert_allclose(value(out, P, 1.6), expected, atol=1e-5)
    complete = lower_spatial(out)
    np.testing.assert_allclose(value(complete, P, 1.6), expected, atol=1e-5)
    assert lower_spatial(complete) is complete


def test_partial_applications_of_transformed_operators_remain_and_are_accepted():
    P = RD(1)
    V = space("operator-input", domain=P, scalar_type=ScalarType.real)
    A = function(OperatorSpace(P, (V, V)), "operator")
    u, v = function(V, "u"), function(V, "v")
    e = action(action(gradient(A, P), 1, u), 0, v)
    assert apply_actions(e) is e
    assert _actions_complete(e)
    assert not _core_normal_form(e)
    assert lower(e) is e


def test_side_pullback_transforms_output_before_deferral_and_keeps_tuple_family():
    import basix
    from test_basix import square

    from uflx_spec_interpreter import side_map
    from uflx_toy.basix_cells import basix_cell

    D, E, _ = square(basix_cell(basix.CellType.triangle))
    sm = side_map(D, E, (E.topology if hasattr(E, "topology") else E).sides[0][0][0])
    assert isinstance(sm, SideMap)
    V = space("tuple-input", domain=D, scalar_type=ScalarType.real)
    A = function(OperatorSpace(D, ((V, V),)), "tuple-operator")
    u, v = function(V, "u"), function(V, "v")
    k = index("argument-family")
    g = (tensor((u, v))[k], v)
    original = pull_back(action(A, 0, g), sm)
    assert not _pull_backs_expanded(original)
    out = expand_pull_backs(original)
    assert isinstance(out, Action)
    assert out.argument is g
    assert out.form is pull_back(A, sm)
    assert out.free_indices == original.free_indices
    assert _pull_backs_expanded(out) and not _core_normal_form(out)
    assert apply_actions(out) is out


def test_pullback_of_bound_form_keeps_binding_entries_in_the_original_space():
    _, _, _, v, A, psi = setup()
    out = expand_pull_backs(pull_back(A, psi))
    assert out.arguments == (v,)
    assert out.argument_spaces == A.argument_spaces
