# Copyright (C) 2026 Jack S. Hale
#
# This file is part of uflx-core-experimental (https://fenics.github.io/uflx/language/)
#
# SPDX-License-Identifier:    MIT
"""Cell-local interpolation with physical components and non-identity geometry."""

import basix
import jax
import numpy as np
import pytest

import uflx_jax_compiler
from uflx.arithmetic import logarithm, num
from uflx.calculus import coordinate
from uflx.forms import function
from uflx.geometry import pull_back, push_forward
from uflx.nodes import EuclideanSpace, Mesh
from uflx.passes import lower
from uflx.tensors import tensor
from uflx_basix.cells import topology
from uflx_basix.interpolation import interpolate
from uflx_basix.spaces import local_space

from .reference import reference_trace


@pytest.mark.parametrize(
    "family",
    [
        basix.ElementFamily.N1E,
        basix.ElementFamily.RT,
        basix.ElementFamily.Regge,
        basix.ElementFamily.HHJ,
    ],
)
def test_curved_reversed_triangle_matches_basix(family):
    T = topology(basix.CellType.triangle)
    (K,) = T.cell_types
    P2 = basix.create_element(basix.ElementFamily.P, K.cell_type, 2)
    E = basix.create_element(family, K.cell_type, 1)
    x = function(local_space(T, P2, shape=(2,)), "x")
    mesh = Mesh(x, EuclideanSpace(2))
    V = local_space(T, E)
    U = local_space(T, P2, shape=V.value_shape)
    u = function(U, "u")
    interpolant = interpolate(u, V, cell=mesh[0])
    assert interpolant == interpolate(push_forward(u, mesh, U.law), V)

    r, s = P2.points.T
    coords = np.column_stack((0.25 * r + s, 2 * r + s + 0.3 * r * s))
    coeffs = np.random.default_rng(52).normal(size=(P2.dim, U.block_size))
    point = np.array([[0.23, 0.37]])

    def geometry(points):
        jacobians = np.array([[[0.25, 1], [2 + 0.3 * s, 1 + 0.3 * r]] for r, s in points])
        return jacobians, np.linalg.det(jacobians), np.linalg.inv(jacobians)

    physical = P2.tabulate(0, E.points)[0, :, :, 0] @ coeffs
    samples = E.pull_back(physical[:, None, :], *geometry(E.points))[:, 0, :]
    dofs = E.interpolation_matrix @ samples.T.ravel()
    reference = dofs @ E.tabulate(0, point)[0, 0]
    local = lower(pull_back(interpolant, T[0]))
    actual_ref = reference_trace(local, x, u, K)(coords, coeffs.ravel(), point[0])
    np.testing.assert_allclose(actual_ref, reference.reshape(V.value_shape), atol=1e-11)
    expected = E.push_forward(reference[None, None, :], *geometry(point)).reshape(V.value_shape)
    value = lower(pull_back(push_forward(interpolant, mesh, V.law), mesh[0]))
    actual = jax.jit(reference_trace(value, x, u, K))(coords, coeffs.ravel(), point[0])
    np.testing.assert_allclose(actual, expected, atol=1e-11)


def test_reference_nedelec_function_uses_its_source_law():
    T = topology(basix.CellType.triangle)
    (K,) = T.cell_types
    P1 = basix.create_element(basix.ElementFamily.P, K.cell_type, 1)
    N1E = basix.create_element(basix.ElementFamily.N1E, K.cell_type, 1)
    x = function(local_space(T, P1, shape=(3,)), "x")
    mesh = Mesh(x, EuclideanSpace(3))
    U = local_space(T, N1E)
    u = function(U, "u")
    V = local_space(T, P1, shape=(3,))
    result = interpolate(u, V, cell=mesh[0])
    assert result == interpolate(push_forward(u, mesh, U.law), V)


def test_ambient_field_on_two_mesh_cells_uses_each_targets_points_and_basis():
    T = topology(basix.CellType.interval, basix.CellType.interval)
    K = T.cell_types[0]
    P1, P2 = (basix.create_element(basix.ElementFamily.P, K.cell_type, p) for p in (1, 2))
    x = function(local_space(T, P1, shape=(1,)), "x")
    mesh = Mesh(x, EuclideanSpace(1))
    X = coordinate(mesh.ambient_domain)[0]
    V = local_space(T, (P1, P2))
    with pytest.raises(ValueError, match="select a cell"):
        interpolate(coordinate(mesh)[0] ** num(2), V)

    coords = np.array([[0.0], [2.0], [2.0], [3.0]])
    for slot, expected in enumerate((2.0, 2.5**2)):
        result = interpolate(X * X, V, cell=mesh[slot])
        value = lower(pull_back(result, T[slot]))
        np.testing.assert_allclose(uflx_jax_compiler.trace(value, x, K)(coords, [0.5]), expected)
        cell = mesh[slot]
        on_cell = V.on_cell(cell)
        assert on_cell is local_space(cell, V.element(slot))
        field = coordinate(mesh)[0] ** num(2)
        restricted = pull_back(field, cell)
        for local in (
            interpolate(field, on_cell),
            pull_back(interpolate(restricted, V), T[slot]),
            interpolate(restricted, local_space(K, V.element(slot))),
        ):
            assert local.domain is K
            np.testing.assert_allclose(
                uflx_jax_compiler.trace(lower(local), x, K)(coords, [0.5]), expected
            )

    with pytest.raises(ValueError, match="different cells"):
        interpolate(pull_back(field, mesh[0]), V.on_cell(mesh[1]))


def test_cell_bound_rotation_supplies_the_piece_on_a_two_triangle_mesh():
    T = topology(basix.CellType.triangle, basix.CellType.triangle)
    cell_type = basix.CellType.triangle
    P1 = basix.create_element(basix.ElementFamily.P, cell_type, 1)
    P2 = basix.create_element(basix.ElementFamily.P, cell_type, 2)
    N1E = basix.create_element(basix.ElementFamily.N1E, cell_type, 1)
    x = function(local_space(T, P1, shape=(2,)), "x")
    mesh = Mesh(x, EuclideanSpace(2))
    cell = mesh[1]
    theta = function(local_space(cell, P2, shape=(2,)), "theta")
    V = local_space(T, N1E)
    result = interpolate(theta, V)
    assert result.domain is T
    result = pull_back(result, T[cell.cell_slot])
    with pytest.raises(ValueError, match="different cells"):
        interpolate(theta, V.on_cell(mesh[0]))

    # Only the second cell has valid geometry. Supply just its six rotation DOFs per component.
    coords = np.array([[np.nan, np.nan]] * 3 + [[0, 0], [2, 0], [2, 1]])
    r, s = P2.points.T
    X = 2 * (r + s)
    nodal_values = np.column_stack((X**2 - s, X))
    for expr in (result, lower(result)):
        value = jax.jit(reference_trace(expr, x, theta, cell.reference_cell))
        np.testing.assert_allclose(value(coords, nodal_values, [0.25, 0.5]), [1.0, 2.5])


def test_interpolation_only_samples_the_selected_cell():
    T = topology(basix.CellType.interval, basix.CellType.interval)
    K = T.cell_types[0]
    P1 = basix.create_element(basix.ElementFamily.P, K.cell_type, 1)
    x = function(local_space(T, P1, shape=(1,)), "x")
    mesh = Mesh(x, EuclideanSpace(1))
    V = local_space(T, P1)
    X = coordinate(mesh.ambient_domain)[0]
    coords = np.array([[1.0], [2.0], [-2.0], [-1.0]])

    # Each expression has finite samples only on its own cell.
    for cell, f in zip(mesh, (logarithm(X), logarithm(-X)), strict=True):
        restriction = interpolate(f, V.on_cell(cell))
        for expression in (restriction, lower(restriction)):
            midpoint = jax.jit(reference_trace(expression, x, K))(coords, np.array([0.5]))
            np.testing.assert_allclose(midpoint, np.log(2) / 2)


def test_constant_ambient_vector_interpolated_into_surface_nedelec():
    T = topology(basix.CellType.triangle)
    (K,) = T.cell_types
    P1 = basix.create_element(basix.ElementFamily.P, K.cell_type, 1)
    N1E = basix.create_element(basix.ElementFamily.N1E, K.cell_type, 1)
    x = function(local_space(T, P1, shape=(3,)), "x")
    mesh = Mesh(x, EuclideanSpace(3))
    f = tensor((num(1), num(2), num(4)))
    V = local_space(T, N1E)
    result = interpolate(f, V, cell=mesh[0])
    assert result.domain == T and result.value_shape == (2,)

    # The tangent plane is spanned by (1, 0, 0) and (0, 1, 1).
    coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 1.0]])
    value = lower(pull_back(push_forward(result, mesh, V.law), mesh[0]))
    np.testing.assert_allclose(
        uflx_jax_compiler.trace(value, x, K)(coords, [0.2, 0.3]), [1, 3, 3], atol=1e-12
    )


def test_blocked_density_dofs_on_a_curved_interval():
    T = topology(basix.CellType.interval)
    (K,) = T.cell_types
    P1 = basix.create_element(basix.ElementFamily.P, K.cell_type, 1, discontinuous=True)
    E = basix.create_custom_element(
        K.cell_type,
        [],
        P1.wcoeffs,
        P1.x,
        P1.M,
        0,
        basix.MapType.L2Piola,
        P1.sobolev_space,
        True,
        1,
        1,
        P1.polyset_type,
    )
    P2 = basix.create_element(basix.ElementFamily.P, K.cell_type, 2)
    x = function(local_space(T, P2, shape=(1,)), "x")
    mesh = Mesh(x, EuclideanSpace(1))
    X = coordinate(mesh)[0]
    field = tensor((X, X * X))
    V = local_space(T, E, shape=(2,))
    interpolant = interpolate(field, V, cell=mesh[0])
    result = lower(pull_back(push_forward(interpolant, mesh, V.law), mesh[0]))

    # F(r) = 2 - r - r². The density DOFs use F'(0) = -1 and F'(1) = -3.
    r = P2.points
    coordinates = 2 - r - r**2
    t = 0.25
    expected = (1 - t) * np.array([2.0, 4.0]) / (1 + 2 * t)
    np.testing.assert_allclose(reference_trace(result, x, K)(coordinates, [t]), expected)
