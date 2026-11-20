"""Shared transformation laws preserve canonical cell geometry and gradient values."""

import numpy as np
import pytest
from testutil import grouped_topology

from uflx_oracle import at, evaluate
from uflx_spec_interpreter import (
    EuclideanSpace,
    ScalarType,
    cell_map,
    expand_pull_backs,
    function,
    gradient,
    jacobian,
    mesh,
    pull_back,
)
from uflx_spec_interpreter._src.algorithms.substitution import substitute
from uflx_spec_interpreter._src.nodes import CellMap, Gradient, TransformationLaw
from uflx_spec_interpreter._src.passes.lowering import lower_spatial
from uflx_spec_interpreter._src.rules.coordinates import transformed
from uflx_toy.cells import reference_simplex
from uflx_toy.spaces import space


def geometry():
    K = reference_simplex(1)
    T = grouped_topology((K,), (((0, 1),),), ())
    V = space("canonical-coordinates", value_shape=(2,), domain=T, scalar_type=ScalarType.real)
    x, y = function(V, "x"), function(V, "y")
    return T, x, y, mesh(x, EuclideanSpace(2))


def test_cell_construction_and_coordinate_replacement_keep_one_representation():
    _T, x, y, D = geometry()
    C = cell_map(D, 0)
    manual = cell_map(D, 0)
    assert isinstance(C, CellMap)
    assert manual is C
    assert manual.parent is D and manual.cell_slot == 0
    assert jacobian(manual) is jacobian(C)
    E = mesh(y, EuclideanSpace(2))
    assert substitute(C, {x: y}) is cell_map(E, 0)
    assert substitute(jacobian(C), {x: y}) is jacobian(cell_map(E, 0))


@pytest.mark.parametrize("vector", [False, True])
def test_gradient_pullback_uses_covariant_law_for_scalar_and_vector_fields(vector):
    _T, x, _, D = geometry()
    u = function(space("physical-field", value_shape=(2,) if vector else (), domain=D), "u")
    C = cell_map(D, 0)
    original = pull_back(gradient(u, D), C)
    actual = expand_pull_backs(original)
    expected = transformed(
        Gradient(expand_pull_backs(pull_back(u, C)), C.reference_cell),
        TransformationLaw(("covariant",), 0),
        C,
    )
    assert actual is expected
    assert actual.value_shape == original.value_shape
    assert actual.free_indices == original.free_indices
    assert actual.argument_spaces == original.argument_spaces
    assert actual.domain is original.domain
    # Straight curve (t, 2t) in R^2: the tangent projection of (1,0) is (1,2)/5.
    # Component two, for a vector field, has ambient gradient (0,1).
    env = {x: lambda X: np.array([X[0], 2 * X[0]])}
    env[u] = (lambda X: np.array([X[0], X[1]])) if vector else (lambda X: X[0])
    indices = {}
    out = lower_spatial(actual)
    value = at(C.reference_cell, np.array([0.3]), lambda: evaluate(out, env, indices)(()))
    desired = np.array([[1, 2], [2, 4]]) / 5 if vector else np.array([1, 2]) / 5
    np.testing.assert_allclose(value, desired, atol=1e-6)
