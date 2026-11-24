"""Metadata and DAG-sharing regressions for lexical alpha comparison."""

import pytest

from uflx._src.algorithms.binding import alpha_equivalent
from uflx.arithmetic import num
from uflx.calculus import integral
from uflx.forms import form, function, rebind
from uflx.geometry import create_mesh
from uflx.nodes import EuclideanSpace

from ..uflx.spaces import space
from .cells import reference_simplex
from .meshes import grouped_topology


def geometry():
    K = reference_simplex(1)
    topology = grouped_topology((K,), (((0, 1),),), ())
    X = space("alpha-coordinates", value_shape=(1,), domain=topology, scalar_type="real")
    x, y = function(X, "x"), function(X, "y")
    V = space("alpha-physical", domain=create_mesh(x, EuclideanSpace(1)), scalar_type="real")
    return x, y, function(V, "c")


def test_geometry_dependent_coefficient_after_public_rebind():
    x, y, c = geometry()
    left = form(c, (x,))
    right = rebind(left, {x: y})
    assert right.expression.space.domain.coordinate_function is y
    assert alpha_equivalent(left, right)
    assert alpha_equivalent(right, left)
    assert rebind(right, {y: x}) is left
    # Without the outer binding correspondence these are different fields.
    assert not alpha_equivalent(left.expression, right.expression)


@pytest.mark.parametrize("used", [False, True])
def test_inner_parameter_space_must_be_independent_of_outer_binding(used):
    x, _, c = geometry()
    with pytest.raises(ValueError, match="argument space depends"):
        form(form(c if used else num(1), (c,)), (x,))


def test_free_identity_and_space_differences_are_preserved():
    x, _, c = geometry()
    left = form(c, (x,))
    other_label = function(c.space, "another-coefficient")
    other_space = function(space("another-space", domain=c.domain, scalar_type="real"), "c")
    assert not alpha_equivalent(left, form(other_label, (x,)))
    assert not alpha_equivalent(left, form(other_space, (x,)))
    # A bound name does not make arbitrary differences in parameter types equal.
    assert not alpha_equivalent(form(c, (c,)), form(other_space, (other_space,)))


def test_parameter_metadata_is_relevant_to_cache_context():
    x, y, c = geometry()
    # The same c is encountered once with x bound and once with x free.
    body = integral(c, c.domain)
    left = form(body, (x,)) + form(body, (y,))
    right_first = rebind(form(body, (x,)), {x: y})
    right = right_first + form(body, (x,))
    assert not alpha_equivalent(left, right)
