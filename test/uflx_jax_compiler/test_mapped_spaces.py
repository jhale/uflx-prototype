# Copyright (C) 2026 Jack S. Hale
#
# This file is part of uflx-core-experimental (https://fenics.github.io/uflx/language/)
#
# SPDX-License-Identifier:    MIT
"""Physical fields share the reference space's coefficient layout."""

import basix
import jax
import numpy as np

from uflx.calculus import coordinate, directional_derivative, integral
from uflx.forms import apply, form, function
from uflx.geometry import create_mesh, euclidean_space, pull_back, push_forward
from uflx.passes import lower
from uflx_basix.cells import create_topology
from uflx_basix.quadrature import apply_quadrature
from uflx_basix.spaces import function_space
from uflx_jax_compiler import trace


def test_embedded_piola_field_holds_dofs_fixed_under_shape_differentiation():
    triangle = basix.CellType.triangle
    T = create_topology(triangle)
    P1 = basix.create_element(basix.ElementFamily.P, triangle, 1)
    N1E = basix.create_element(basix.ElementFamily.N1E, triangle, 1)
    X = function_space(T, P1, shape=(3,))
    x, h = function(X, "x"), function(X, "h")
    mesh = create_mesh(x, euclidean_space(3))
    U = function_space(T, N1E)
    V = function_space(mesh, N1E)
    u = function(V, "u")
    assert u.domain == mesh and u.value_shape == (3,)
    u_ref = function(U, "u")
    physical = push_forward(u_ref, mesh, U.law)
    rule = basix.make_quadrature(triangle, 2)
    energy = integral(u @ u, mesh)
    oracle = trace(apply_quadrature(lower(integral(physical @ physical, mesh)), *rule), x, u_ref)
    derivative = directional_derivative(energy, x, h)
    actual = trace(apply_quadrature(lower(derivative), *rule), x, h, u)
    vertices = np.array([[1.0, 0.0, 2.0], [2.0, 0.0, 3.0], [0.0, 3.0, 4.0]])
    direction = np.array([[0.2, 0.0, 0.1], [1.0, -0.7, 0.1], [0.8, 0.4, -0.2]])
    dofs = np.array([0.2, 0.7, -1.3])
    _, expected = jax.jvp(lambda coords: oracle(coords, dofs), (vertices,), (direction,))
    np.testing.assert_allclose(actual(vertices, direction, dofs), expected, atol=1e-10)


def test_substituting_a_physical_expression_after_cell_lowering():
    triangle = basix.CellType.triangle
    T = create_topology(triangle)
    P1 = basix.create_element(basix.ElementFamily.P, triangle, 1)
    RT = basix.create_element(basix.ElementFamily.RT, triangle, 1)
    x = function(function_space(T, P1, shape=(2,)), "x")
    mesh = create_mesh(x, euclidean_space(2))
    u = function(function_space(mesh, RT), "u")
    slot = form(lower(pull_back(u, mesh[0])), (u,))
    point = coordinate(mesh)
    evaluate = trace(lower(apply(slot, (point,))), x, T.cell_types[0])
    coords = np.array([[0.5, 1.0], [1.0, 4.0], [2.0, 1.0]])
    r, s = 0.2, 0.3
    np.testing.assert_allclose(evaluate(coords, [r, s]), [1 - r - s, r, s] @ coords)
