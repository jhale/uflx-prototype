"""Fixed matrix sizes and polynomial determinant derivatives at singular points."""

import numpy as np
import pytest

from uflx._src.nodes import Inverse
from uflx.arithmetic import num
from uflx.calculus import directional_derivative
from uflx.forms import function
from uflx.passes import apply_derivatives
from uflx.tensors import component_tensor, determinant, index, inverse, tensor
from uflx_basix.runtime import runtime_array, runtime_extent
from uflx_basix.spaces import space
from uflx_oracle import evaluate

from ..testutil import contains


@pytest.mark.parametrize("operation", [determinant, inverse])
@pytest.mark.parametrize("shape_kind", ["square", "rows", "columns"])
def test_matrix_operations_reject_symbolic_dimensions(operation, shape_kind):
    n = runtime_extent("matrix-size")
    shape = {"square": (n, n), "rows": (n, 2), "columns": (2, n)}[shape_kind]
    with pytest.raises(ValueError, match="fixed integers"):
        operation(runtime_array("matrix", shape))


@pytest.mark.parametrize("operation", [determinant, inverse])
def test_fixed_matrices_retain_symbolic_family_indices(operation):
    q, i = runtime_extent("matrix-family"), index("i")
    matrices = runtime_array("matrices", (q, 2, 2))
    result = operation(matrices[i])
    assert result.free_indices == ((i, q),)
    assert result.value_shape == (() if operation is determinant else (2, 2))


@pytest.mark.parametrize("n", [0, 1, 2, 3])
@pytest.mark.parametrize("scalar_type", ["real", "complex"])
def test_determinant_derivative_at_singular_matrix(n, scalar_type):
    V = space("determinant-matrix", (n, n), scalar_type=scalar_type)
    A, H = function(V, "A"), function(V, "H")
    derivative = apply_derivatives(directional_derivative(determinant(A), A, A, H))
    assert not contains(derivative, lambda node: isinstance(node, Inverse))
    assert derivative.scalar_type == scalar_type
    point = np.eye(n)
    if n:
        point[0, 0] = 0
    direction = np.arange(1, n * n + 1).reshape(n, n).astype(float)
    if scalar_type == "complex":
        direction = direction * (1 + 2j)
    # At diag(0, 1, ...), only the (0, 0) cofactor is nonzero.
    expected = direction[0, 0] if n else 0
    np.testing.assert_allclose(evaluate(derivative, {A: point, H: direction})(()), expected)


def test_determinant_derivative_uses_point_and_preserves_its_family():
    V = space("determinant-scalar", scalar_type="real")
    t = function(V, "t")
    A = tensor((tensor((t, num(0))), tensor((num(0), t))))
    i = index("point")
    points = tensor((num(0), num(3)))[i]
    derivative = apply_derivatives(
        component_tensor(directional_derivative(determinant(A), t, points, num(1)), (i,))
    )
    np.testing.assert_allclose(evaluate(derivative, {})(()), [0, 6])


def test_second_determinant_derivative_at_zero():
    t = function(space("determinant-second", scalar_type="real"), "t")
    A = tensor((tensor((t, num(0))), tensor((num(0), t))))
    first = directional_derivative(determinant(A), t, t, num(1))
    second = apply_derivatives(directional_derivative(first, t, num(0), num(1)))
    assert not contains(second, lambda node: isinstance(node, Inverse))
    np.testing.assert_allclose(evaluate(second, {})(()), 2)


def test_determinant_derivative_matches_finite_difference():
    V = space("determinant-dense", (3, 3), scalar_type="complex")
    A, H = function(V, "A"), function(V, "H")
    derivative = apply_derivatives(directional_derivative(determinant(A), A, A, H))
    rng = np.random.default_rng(17)
    point = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
    direction = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
    step = 1e-5
    expected = (
        np.linalg.det(point + step * direction) - np.linalg.det(point - step * direction)
    ) / (2 * step)
    np.testing.assert_allclose(
        evaluate(derivative, {A: point, H: direction})(()), expected, rtol=1e-8
    )
