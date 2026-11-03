"""Shared transformation laws preserve canonical cell geometry and gradient values."""

import numpy as np
import pytest

from uflx_oracle import at, evaluate
from uflx_spec_interpreter import (
    RD,
    ScalarType,
    cell_map,
    expand_pull_backs,
    function,
    gradient,
    image,
    jacobian,
    pull_back,
)
from uflx_spec_interpreter._src.domains import cell_domain, cell_of
from uflx_spec_interpreter._src.nodes import CellMap, Gradient, TransformationLaw
from uflx_spec_interpreter._src.passes.gradients import lower as lower_spatial
from uflx_spec_interpreter._src.passes.pullbacks import transformed
from uflx_spec_interpreter._src.substitution import substitute
from uflx_toy.cells import reference_simplex
from uflx_toy.spaces import space
from uflx_toy.topology import static_topology


def geometry():
    K = reference_simplex(1)
    T = static_topology((K,), (((0, 1),),), ())
    V = space("canonical-coordinates", value_shape=(2,), domain=T, scalar_type=ScalarType.real)
    x, y = function(V, "x"), function(V, "y")
    return T, x, y, image(x, RD(2))


def test_cell_construction_and_coordinate_replacement_keep_one_representation():
    T, x, y, D = geometry()
    C = cell_domain(D, 0)
    manual = cell_map(D, 0)
    assert isinstance(C, CellMap)
    assert manual is C
    assert cell_of(manual) == (0, cell_map(T, 0).entry)
    assert jacobian(manual) is jacobian(C)
    E = image(y, RD(2))
    assert substitute(C, {x: y}) is cell_domain(E, 0)
    assert substitute(jacobian(C), {x: y}) is jacobian(cell_domain(E, 0))


@pytest.mark.parametrize("vector", [False, True])
def test_gradient_pullback_uses_covariant_law_for_scalar_and_vector_fields(vector):
    T, x, _, D = geometry()
    u = function(space("physical-field", value_shape=(2,) if vector else (), domain=D), "u")
    C = cell_domain(D, 0)
    original = pull_back(gradient(u, D), C)
    actual = expand_pull_backs(original)
    expected = transformed(
        Gradient(expand_pull_backs(pull_back(u, C)), C.parameter_domain),
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
    indices = {cell_map(T, 0).entry: 0}
    out = lower_spatial(actual)
    value = at(C.parameter_domain, np.array([0.3]), lambda: evaluate(out, env, indices)(()))
    desired = np.array([[1, 2], [2, 4]]) / 5 if vector else np.array([1, 2]) / 5
    np.testing.assert_allclose(value, desired, atol=1e-6)
