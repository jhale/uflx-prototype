# Copyright (C) 2026 Jack S. Hale
#
# This file is part of uflx-core-experimental (https://fenics.github.io/uflx/language/)
#
# SPDX-License-Identifier:    MIT

import basix
import numpy as np
import pytest

import uflx_jax_compiler
from uflx._src.passes import integrals
from uflx._src.passes.contracts import in_core_normal_form
from uflx._src.passes.lowering import lower_spatial
from uflx.calculus import gradient, integral
from uflx.forms import function
from uflx.geometry import pull_back
from uflx.passes import expand_arithmetic
from uflx_basix.cells import cell, topology
from uflx_basix.spaces import local_space

from ..uflx_basix.meshes import mixed, push_forward, side_map, square
from .quadrature import integrate_at

P = basix.ElementFamily.P
# The vertex coordinates of the square of test_basix and of its mixed mesh.
SQUARE = [(0, 0), (1, 0), (1, 1), (0, 1)]
MIXED = [(0, 0), (0.5, 0), (1, 0), (0, 1), (0.5, 1), (1, 1)]


def interpolate(V, f, coordinates):
    # The coefficients of the interpolant of f, at the points of the element in each cell.
    U = np.zeros(V.num_dofs)
    T = V.topology
    for j, vertices in enumerate(T.cells):
        E, cell_type = V.element(j), T.cell_types[j].cell_type
        shapes = basix.create_element(P, cell_type, 1).tabulate(0, E.points)[0, :, :, 0]
        x = shapes @ np.array([coordinates[v] for v in vertices], dtype=float)
        U[list(V.cell_dofs[j])] = [f(p) for p in x]
    return U


def value(e, env):
    out = lower_spatial(integrals.expand_integrals(e))
    out = expand_arithmetic(out)
    assert in_core_normal_form(out)
    before, after = integrate_at(e, env, degree=12), integrate_at(out, env, degree=12)
    assert np.isclose(before, after, rtol=1e-6, atol=1e-9), (before, after)
    return np.real(after)


def jump_squared(u, D, E):
    plus, minus = side_map(D, E, "+"), side_map(D, E, "-")
    jump = pull_back(u, plus) - pull_back(u, minus)
    return integral(jump * jump, E)


@pytest.mark.parametrize("degree", [1, 2])
def test_an_interpolated_polynomial_integrates_exactly(degree):
    D, _, _, geometry = square(cell(basix.CellType.triangle))
    V = local_space(D.topology, basix.create_element(P, basix.CellType.triangle, degree))
    uT = function(V, "u")
    u = push_forward(uT, D)
    f = (lambda x: 1 + x[0] + 2 * x[1]) if degree == 1 else (lambda x: x[0] ** 2 + x[0] * x[1])
    env = geometry | {uT: interpolate(V, f, SQUARE)}
    expected = 2.5 if degree == 1 else 1 / 3 + 1 / 4
    assert np.isclose(value(integral(u, D), env), expected)
    assert np.isclose(value(integral(gradient(u, D)[1], D), env), 2 if degree == 1 else 0.5)


def test_matching_cell_local_coefficients_make_traces_continuous():
    D, E, _, geometry = square(cell(basix.CellType.triangle))
    rng = np.random.default_rng(3)
    for degree in (1, 2):
        V = local_space(D.topology, basix.create_element(P, basix.CellType.triangle, degree))
        uT = function(V, "u")
        coefficients = geometry | {uT: interpolate(V, lambda x: np.cos(x[0] + 2 * x[1]), SQUARE)}
        assert np.isclose(
            value(jump_squared(push_forward(uT, D), D, E), coefficients), 0, atol=1e-10
        )
    W = local_space(
        D.topology,
        basix.create_element(P, basix.CellType.triangle, 1, discontinuous=True),
    )
    wT = function(W, "w")
    w = push_forward(wT, D)
    assert value(jump_squared(w, D, E), geometry | {wT: rng.normal(size=W.num_dofs)}) > 1e-3
    # A discontinuous space that interpolates a continuous function has no jump either.
    env = geometry | {wT: interpolate(W, lambda x: 1 + x[0] - x[1], SQUARE)}
    assert np.isclose(value(jump_squared(w, D, E), env), 0, atol=1e-10)


def test_a_continuous_space_on_triangles_and_a_quadrilateral():
    D, E, _, geometry = mixed()
    elements = tuple(basix.create_element(P, K.cell_type, 2) for K in D.topology.cell_types)
    V = local_space(D.topology, elements)
    uT = function(V, "u")
    u = push_forward(uT, D)
    env = geometry | {uT: interpolate(V, lambda x: x[0] ** 2 + x[0] * x[1] - x[1], MIXED)}
    assert np.isclose(value(integral(u, D), env), 1 / 3 + 1 / 4 - 1 / 2)
    assert np.isclose(value(jump_squared(u, D, E), env), 0, atol=1e-10)


@pytest.mark.parametrize("shape", [(2,), (2, 2)])
def test_blocked_coefficients_accept_one_value_per_scalar_dof(shape):
    triangle = basix.CellType.triangle
    T = topology(triangle, triangle)
    p1 = basix.create_element(P, triangle, 1)
    V = local_space(T, p1, shape=shape)
    u = function(V, "u")
    values = np.arange(V.num_dofs, dtype=float).reshape(-1, *shape)
    sample = uflx_jax_compiler.trace(pull_back(u, T[1]), u, T[1].reference_cell)

    for coefficients in (values, values.ravel()):
        result = sample(coefficients, p1.points[1])
        np.testing.assert_allclose(result, values[p1.dim + 1])

    with pytest.raises(ValueError, match="expected coefficient shape"):
        sample(values.swapaxes(0, 1), p1.points[1])
