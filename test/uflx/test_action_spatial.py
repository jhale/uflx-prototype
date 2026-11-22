"""Spatial operators transform a form's output without transforming its arguments."""

from __future__ import annotations

import numpy as np
import pytest

from uflx._src.nodes import AbstractDomain, AbstractFunctionSpace, Apply, SideMap
from uflx._src.passes.applications import applications_expanded
from uflx._src.passes.contracts import core_normal_form
from uflx._src.passes.lowering import lower_spatial
from uflx._src.passes.pullbacks import pull_backs_expanded
from uflx.arithmetic import num
from uflx.calculus import coordinate, gradient
from uflx.forms import apply, form, function
from uflx.geometry import domain_map, pull_back
from uflx.nodes import EuclideanSpace, ScalarType
from uflx.passes import apply_gradients, expand_applications, expand_pull_backs, lower
from uflx.tensors import index, tensor
from uflx_basix.cells import reference_simplex
from uflx_basix.spaces import space
from uflx_oracle import at, evaluate


class OperatorSpace(AbstractFunctionSpace):
    """A library-supplied operator with field-valued output and specified slots."""

    identifier = "test.OperatorSpace"
    _domain: AbstractDomain
    _argument_spaces: tuple
    value_shape = ()
    scalar_type = ScalarType.real

    @property
    def domain(self):
        return self._domain

    @property
    def argument_spaces(self):
        return self._argument_spaces


def setup():
    P, K = EuclideanSpace(1), reference_simplex(1)
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
    e = pull_back(apply(gradient(A, P), (g,)), psi)
    out = expand_pull_backs(e)
    assert isinstance(out, Apply)
    assert out.arguments[0] == g and out.arguments[0].domain is P
    assert out.argument_spaces == e.argument_spaces
    np.testing.assert_allclose(value(out, K, 0.3), [3 * 1.6**2], atol=1e-6)
    np.testing.assert_allclose(value(out, K, 0.3), value(e, K, 0.3), atol=1e-6)
    complete = lower(e)
    assert not core_normal_form(complete)
    assert lower_spatial(complete) is complete
    np.testing.assert_allclose(value(complete, K, 0.3), [3 * 1.6**2], atol=1e-5)


@pytest.mark.parametrize("retained", [False, True])
def test_gradient_does_not_count_argument_variation_twice(retained):
    P, _, x, _, A, _ = setup()
    f = gradient(A, P) if retained else A
    e = gradient(apply(f, (x * x,)), P)
    out = apply_gradients(e)
    expected = [[6 * 1.6]] if retained else [3 * 1.6**2]
    np.testing.assert_allclose(value(out, P, 1.6), expected, atol=1e-5)
    complete = lower_spatial(out)
    np.testing.assert_allclose(value(complete, P, 1.6), expected, atol=1e-5)
    assert lower_spatial(complete) is complete


def test_partial_applications_of_transformed_operators_remain_and_are_accepted():
    P = EuclideanSpace(1)
    V = space("operator-input", domain=P, scalar_type=ScalarType.real)
    A = function(OperatorSpace(P, (V, V)), "operator")
    u, v = function(V, "u"), function(V, "v")
    e = apply(gradient(A, P), (v, u))
    assert expand_applications(e) is e
    assert applications_expanded(e)
    assert not core_normal_form(e)
    assert lower(e) is e


def test_side_pullback_transforms_output_before_deferral_and_keeps_tuple_family():
    import basix

    from uflx_basix.basix_cells import basix_cell

    from ..testutil import side_map
    from ..uflx_basix.test_basix import square

    D, E, _ = square(basix_cell(basix.CellType.triangle))
    sm = side_map(D, E)
    assert isinstance(sm, SideMap)
    V = space("tuple-input", domain=D, scalar_type=ScalarType.real)
    A = function(OperatorSpace(D, ((V, V),)), "tuple-operator")
    u, v = function(V, "u"), function(V, "v")
    k = index("argument-family")
    g = (tensor((u, v))[k], v)
    original = pull_back(apply(A, (g,)), sm)
    assert not pull_backs_expanded(original)
    out = expand_pull_backs(original)
    assert isinstance(out, Apply)
    assert out.arguments[0] == g
    assert out.form is pull_back(A, sm)
    assert out.free_indices == original.free_indices
    assert pull_backs_expanded(out) and not core_normal_form(out)
    assert expand_applications(out) is out


def test_pullback_of_bound_form_keeps_binding_entries_in_the_original_space():
    _, _, _, v, A, psi = setup()
    out = expand_pull_backs(pull_back(A, psi))
    assert out.arguments == (v,)
    assert out.argument_spaces == A.argument_spaces
