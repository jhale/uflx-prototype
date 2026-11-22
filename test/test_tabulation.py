"""Numerical cleanup of finite-element basis tables."""

import basix
import numpy as np
import pytest

from uflx_compiler.tabulation import basis, clamp_table


def test_clamping_preserves_other_values_and_input():
    values = np.array([[-1 - 2e-16, 4e-17, 1 + 2e-16], [0.25, 1e-7, 1 + 2e-6]])
    original = values.copy()
    np.testing.assert_array_equal(clamp_table(values), [[-1, 0, 1], [0.25, 1e-7, 1 + 2e-6]])
    np.testing.assert_array_equal(values, original)
    np.testing.assert_array_equal(clamp_table(values, rtol=0, atol=0), original)


@pytest.mark.parametrize("direction, expected", [(0, [-1, 1, 0]), (1, [-1, 0, 1])])
def test_affine_triangle_gradients_are_exact(direction, expected):
    element = basix.create_element(basix.ElementFamily.P, basix.CellType.triangle, 1)
    points = np.array([[[0.2, 0.3], [0.1, 0.1]], [[0.4, 0.2], [0.3, 0.5]]])
    result = basis(element, (direction,), points)
    np.testing.assert_array_equal(result, np.broadcast_to(expected, (2, 2, 3)))


def test_quadratic_nodal_values_are_exact_without_clamping_interior_values():
    element = basix.create_element(
        basix.ElementFamily.P, basix.CellType.triangle, 2, basix.LagrangeVariant.equispaced
    )
    np.testing.assert_array_equal(basis(element, (), element.points), np.eye(element.dim))
    points = np.array([[0.21, 0.32], [0.11, 0.17]])
    expected = element.tabulate(0, points)[0, :, :, 0]
    np.testing.assert_array_equal(basis(element, (), points), expected)
