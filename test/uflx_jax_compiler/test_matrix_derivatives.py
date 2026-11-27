# Copyright (C) 2026 Jack S. Hale
#
# This file is part of uflx-core-experimental (https://fenics.github.io/uflx/language/)
#
# SPDX-License-Identifier:    MIT
"""Fixed matrix sizes and polynomial determinant derivatives at singular points."""

import jax
import numpy as np
import pytest

import uflx_jax_compiler
from uflx._src.nodes import Inverse
from uflx.arithmetic import num
from uflx.calculus import directional_derivative
from uflx.forms import function
from uflx.passes import apply_derivatives, lower
from uflx.tensors import component_tensor, determinant, index, inverse, tensor
from uflx_jax_compiler.passes import lower as compiler_lower

from ..testutil import contains
from ..uflx.spaces import space


@pytest.mark.parametrize("n", [0, 1, 2, 3])
@pytest.mark.parametrize("scalar_type", ["real", "complex"])
def test_determinant_derivative_at_singular_matrix(n, scalar_type):
    V = space("determinant-matrix", None, (n, n), scalar_type=scalar_type)
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
    np.testing.assert_allclose(
        uflx_jax_compiler.trace(derivative, A, H)(point, direction), expected
    )


def test_determinant_derivative_uses_point_and_preserves_its_family():
    V = space("determinant-scalar", None, scalar_type="real")
    t = function(V, "t")
    A = tensor((tensor((t, num(0))), tensor((num(0), t))))
    i = index("point")
    points = tensor((num(0), num(3)))[i]
    derivative = apply_derivatives(
        component_tensor(directional_derivative(determinant(A), t, points, num(1)), (i,))
    )
    np.testing.assert_allclose(uflx_jax_compiler.trace(derivative)(), [0, 6])


def test_second_determinant_derivative_at_zero():
    t = function(space("determinant-second", None, scalar_type="real"), "t")
    A = tensor((tensor((t, num(0))), tensor((num(0), t))))
    first = directional_derivative(determinant(A), t, t, num(1))
    second = apply_derivatives(directional_derivative(first, t, num(0), num(1)))
    assert not contains(second, lambda node: isinstance(node, Inverse))
    np.testing.assert_allclose(uflx_jax_compiler.trace(second)(), 2)


def test_determinant_derivative_matches_finite_difference():
    V = space("determinant-dense", None, (3, 3), scalar_type="complex")
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
        uflx_jax_compiler.trace(derivative, A, H)(point, direction), expected, rtol=1e-8
    )


@pytest.mark.parametrize(
    ("point", "cofactors"),
    [
        (np.empty((0, 0)), np.empty((0, 0))),
        ([[0.0]], [[1.0]]),
        ([[0.0, 0.0], [0.0, 3.0]], [[3.0, 0.0], [0.0, 0.0]]),
        ([[1.0, 2.0], [3.0, 4.0]], [[4.0, -3.0], [-2.0, 1.0]]),
        ([[1 + 2j, 2], [3j, -4j]], [[-4j, -3j], [-2, 1 + 2j]]),
    ],
    ids=["empty", "singleton", "singular", "nonsingular", "complex"],
)
def test_determinant_jvp_and_gradient(point, cofactors):
    point = np.asarray(point)
    complex_input = np.iscomplexobj(point)
    A = function(
        space("matrix", None, point.shape, scalar_type="complex" if complex_input else "real"), "A"
    )
    expr = determinant(A)

    def evaluate(matrix):
        return uflx_jax_compiler.trace(expr, A)(matrix)

    direction = np.arange(2, 2 + point.size, dtype=point.dtype).reshape(point.shape)
    grad = jax.jit(jax.grad(evaluate, holomorphic=complex_input))(point)
    _, tangent = jax.jvp(evaluate, (point,), (direction,))
    np.testing.assert_allclose(tangent, np.sum(cofactors * direction))
    np.testing.assert_allclose(grad, cofactors)


def test_determinant_gradient_derivative_at_a_singular_minor():
    A = function(space("matrix", None, (3, 3), scalar_type="real"), "A")
    expr = determinant(A)

    def evaluate(matrix):
        return uflx_jax_compiler.trace(expr, A)(matrix)

    # At diag(2t, 3, 0) the only nonzero cofactor is the final diagonal entry, 6t.
    point = np.diag([0.0, 3.0, 0.0])
    direction = np.diag([2.0, 0.0, 0.0])
    _, mixed = jax.jvp(jax.grad(evaluate), (point,), (direction,))
    np.testing.assert_allclose(mixed, np.diag([0.0, 0.0, 6.0]))


@pytest.mark.parametrize("n", [2, 3])
@pytest.mark.parametrize("scalar_type", ["real", "complex"])
def test_small_inverse_under_vmap_and_ad(n, scalar_type):
    A = function(space("inverse-matrix", None, (n, n), scalar_type=scalar_type), "A")
    kernel = uflx_jax_compiler.trace(inverse(A), A)
    rng = np.random.default_rng(15)
    matrix = rng.normal(size=(n, n)) + n * np.eye(n)
    direction = rng.normal(size=(n, n))
    if scalar_type == "complex":
        matrix = matrix + 1j * rng.normal(size=(n, n))
        direction = direction + 1j * rng.normal(size=(n, n))
    # Permuting the rows gives each mapped item a different first pivot.
    batch = np.stack([np.roll(matrix, k, axis=0) for k in range(n)])
    directions = np.broadcast_to(direction, batch.shape)
    inv = np.linalg.inv(batch)
    mapped = jax.vmap(kernel)
    values, tangents = jax.jit(lambda a, h: jax.jvp(mapped, (a,), (h,)))(batch, directions)
    np.testing.assert_allclose(values, inv, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(tangents, -inv @ directions @ inv, rtol=1e-12, atol=1e-12)
    _, pullback = jax.vjp(mapped, batch)
    transpose = inv.swapaxes(-1, -2)
    np.testing.assert_allclose(pullback(directions)[0], -transpose @ directions @ transpose)


@pytest.mark.parametrize("n", [2, 3])
def test_small_inverse_at_extreme_scales(n):
    A = function(space("scaled-matrix", None, (n, n), scalar_type="real"), "A")
    kernel = jax.jit(jax.vmap(uflx_jax_compiler.trace(inverse(A), A)))
    # The inverse is representable even when a determinant overflows or underflows.
    scales = np.array([1e200, 1e-200])[:, None, None]
    matrix = (np.eye(n) + np.triu(np.ones((n, n))))[::-1]
    result = kernel(scales * matrix) * scales
    np.testing.assert_allclose(result, np.broadcast_to(np.linalg.inv(matrix), result.shape))


@pytest.mark.parametrize("n", [2, 3])
def test_lower_differentiates_small_determinant_at_singular_point(n):
    from uflx.nodes import Determinant

    V = space("small-det", None, (n, n), scalar_type="real")
    A, H, K = (function(V, name) for name in ("A", "H", "K"))
    det = determinant(A)
    first = directional_derivative(det, A, H)
    tangent = directional_derivative(first, A, K)
    assert contains(lower(tangent), lambda node: isinstance(node, Determinant))
    lowered = compiler_lower(tangent)
    assert not contains(lowered, lambda node: isinstance(node, Determinant))
    assert compiler_lower(det) == det
    # Perturb the final 2x2 block in dense directions at a singular point.
    point = np.eye(n)
    point[-2:, -2:] = 0
    h, k = np.zeros_like(point), np.zeros_like(point)
    h[-2:, -2:] = [[1, 2], [3, 4]]
    k[-2:, -2:] = [[5, 6], [7, 8]]
    value = uflx_jax_compiler.trace(lowered, A, H, K)(point, h, k)
    np.testing.assert_allclose(value, 1 * 8 + 4 * 5 - 2 * 7 - 3 * 6)
