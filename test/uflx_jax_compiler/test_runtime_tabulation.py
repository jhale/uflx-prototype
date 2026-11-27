# Copyright (C) 2026 Jack S. Hale
#
# This file is part of uflx-core-experimental (https://fenics.github.io/uflx/language/)
#
# SPDX-License-Identifier:    MIT
"""Compiler basis-table inputs for reference-cell kernels."""

import basix
import jax
import numpy as np
import pytest

import uflx_jax_compiler
from uflx.arithmetic import num
from uflx.calculus import coordinate, gradient, integral
from uflx.forms import form, function
from uflx.geometry import domain_map, pull_back, push_forward
from uflx.nodes import EuclideanSpace, Mesh
from uflx.passes import lower
from uflx_basix.cells import cell, create_topology
from uflx_basix.nodes import BasixBasis
from uflx_basix.quadrature import apply_quadrature
from uflx_basix.spaces import function_space
from uflx_extensions.external_operators import external_operator
from uflx_extensions.runtime_arrays import runtime_array, runtime_extent
from uflx_jax_compiler import basis_tables

from ..uflx_extensions.operations import Energy
from .reference import reference_trace


def test_mass_matrix_and_action_use_native_arrays(monkeypatch):
    interval = cell(basix.CellType.interval)
    p2 = basix.create_element(basix.ElementFamily.P, interval.cell_type, 2)
    space = function_space(interval, p2)
    trial, test = (function(space, name) for name in ("u", "v"))
    mass = integral(trial * test, interval)
    matrix, action = form(mass, (test, trial)), form(mass, (test,))
    (table,) = basis_tables(matrix)
    assert table.element == p2 and table.order == 0 and table.point_source == mass
    points, weights = basix.make_quadrature(interval.cell_type, 4)
    values = p2.tabulate(0, points)
    # Deliberately different rows detect any resampling or reconstruction of local DOFs.
    values = values + 0.3
    basis = values[0, :, :, 0]
    expected = basis.T @ (weights[:, None] * basis)

    def unexpected(*args):
        pytest.fail("the supplied basis table should suffice")

    monkeypatch.setattr(type(p2), "tabulate", unexpected)
    compiled = uflx_jax_compiler.trace(matrix, mass, table)
    assert compiled.inputs == (mass, table)
    np.testing.assert_allclose(jax.jit(compiled)((points, weights), values), expected)
    coeffs = np.array([2.0, -1.0, 3.0])
    kernel = uflx_jax_compiler.trace(action, trial, mass, table)
    fixed_basis = jax.jit(lambda coeffs, rule: kernel(coeffs, rule, values))
    np.testing.assert_allclose(fixed_basis(coeffs, (points, weights)), expected @ coeffs)
    with pytest.raises(ValueError, match="floating Basix table with shape"):
        compiled((points, weights), values[:, :-1])
    with pytest.raises(ValueError, match="floating Basix table with shape"):
        compiled((points, weights), values.astype(int))


def test_curved_poisson_uses_one_table_for_fields_and_geometry():
    triangle = basix.CellType.triangle
    topology_ = create_topology(triangle)
    p2 = basix.create_element(basix.ElementFamily.P, triangle, 2)
    coords = function(function_space(topology_, p2, shape=(2,)), "x")
    mesh = Mesh(coords, EuclideanSpace(2))
    space = function_space(topology_, p2)
    trial, test = (function(space, name) for name in ("u", "v"))
    gu, gv = (gradient(push_forward(f, mesh, space.law), mesh) for f in (trial, test))
    stiffness = lower(integral(gu @ gv, mesh))
    matrix = form(stiffness, (test, trial))
    (table,) = basis_tables(matrix)
    assert table.element == p2 and table.order == 1
    s, t = p2.points.T
    vertices = np.column_stack((s, t + 0.4 * s * t))
    points, weights = basix.make_quadrature(triangle, 2)
    ordinary = reference_trace(matrix, coords, stiffness)
    kernel = uflx_jax_compiler.trace(matrix, coords, stiffness, table)
    compiled = jax.jit(kernel).lower(vertices, (points, weights), p2.tabulate(1, points)).compile()
    for p, w in ((points, weights), (points * 0.8, weights[::-1] * 0.7)):
        values = p2.tabulate(1, p)
        np.testing.assert_allclose(
            compiled(vertices, (p, w), values), ordinary(vertices, (p, w)), atol=2e-13
        )
    direction = np.cos(vertices)
    actual = jax.jvp(
        lambda x: kernel(x, (points, weights), p2.tabulate(1, points)), (vertices,), (direction,)
    )[1]
    expected = jax.jvp(lambda x: ordinary(x, (points, weights)), (vertices,), (direction,))[1]
    np.testing.assert_allclose(actual, expected, atol=2e-13)


def test_independent_integrals_can_start_with_identical_rules():
    triangle = cell(basix.CellType.triangle)
    element = basix.create_element(basix.ElementFamily.P, triangle.cell_type, 2)
    v = function(function_space(triangle, element), "v")
    load = integral(v, triangle)
    mixed = integral(v + gradient(gradient(v, triangle), triangle)[0, 1], triangle)
    linear = form(load + mixed, (v,))
    first, second = basis_tables(linear)
    assert (first.point_source, first.order, second.point_source, second.order) == (
        load,
        0,
        mixed,
        2,
    )
    points, weights = basix.make_quadrature(triangle.cell_type, 2)
    values = element.tabulate(2, points)
    left, right = 2 * values[:1], 3 * values
    kernel = jax.jit(uflx_jax_compiler.trace(linear, load, mixed, first, second))
    for lw, rw in ((weights, weights), (weights * 2, weights[::-1])):
        actual = kernel((points, lw), (points, rw), left, right)
        expected = lw @ left[0, :, :, 0] + rw @ (
            right[0, :, :, 0] + right[basix.index(1, 1), :, :, 0]
        )
        np.testing.assert_allclose(actual, expected, atol=1e-13)
    derivative = jax.grad(lambda a: kernel((points, weights), (points, weights), a, right).sum())(
        left
    )
    np.testing.assert_allclose(derivative[0, :, :, 0], np.tile(weights[:, None], (1, element.dim)))
    empty = uflx_jax_compiler.trace(form(load, (v,)), load, first)
    np.testing.assert_allclose(empty((points[:0], weights[:0]), left[:, :0]), np.zeros(element.dim))


def test_whole_field_operator_and_callable_keep_their_sampling_semantics():
    interval = cell(basix.CellType.interval)
    p1 = basix.create_element(basix.ElementFamily.P, interval.cell_type, 1)
    u = function(function_space(interval, p1), "u")
    energy = external_operator(Energy(), (u, num(1)))
    assert basis_tables(integral(energy, interval)) == ()
    total = integral(u + energy, interval)
    (table,) = basis_tables(total)
    points, weights = basix.make_quadrature(interval.cell_type, 2)
    rule = (points, weights)
    basis = 2 * p1.tabulate(0, points)
    take_sample = lambda field, scale: field(np.array([0.25])) * scale
    evaluate = uflx_jax_compiler.trace(total, total, table, u, Energy())
    np.testing.assert_allclose(evaluate(rule, basis, np.array([0.0, 1.0]), take_sample), 1.25)
    np.testing.assert_allclose(
        evaluate(rule, basis, lambda p: p[0] ** 2, take_sample), 1 / 3 + 0.25**2
    )


def test_tabulations_are_local_to_slot_and_sampling_coordinate():
    triangle = cell(basix.CellType.triangle)
    topology_ = create_topology(triangle.cell_type, triangle.cell_type)
    first, second = (
        basix.create_element(basix.ElementFamily.P, triangle.cell_type, 1, discontinuous=flag)
        for flag in (False, True)
    )
    space = function_space(topology_, (first, second))
    v = function(space, "v")
    load = integral(pull_back(v, topology_[0]) + pull_back(v, topology_[1]), triangle)
    linear = form(load, (v,))
    ltab, rtab = basis_tables(linear)
    assert (ltab.element, rtab.element) == (first, second)
    points, weights = basix.make_quadrature(triangle.cell_type, 2)
    left, right = 2 * first.tabulate(0, points), 3 * second.tabulate(0, points)
    actual = jax.jit(uflx_jax_compiler.trace(linear, load, ltab, rtab))(
        (points, weights), left, right
    )
    expected = np.concatenate([weights @ data[0, :, :, 0] for data in (left, right)])
    np.testing.assert_allclose(actual, expected, atol=1e-14)

    rt = basix.create_element(basix.ElementFamily.RT, triangle.cell_type, 1)
    f = function(function_space(triangle, rt), "f")
    mapped = pull_back(f, domain_map(triangle, coordinate(triangle) * num(0.5), triangle))
    total = integral(f + mapped, triangle)
    (operand,) = basis_tables(total)
    coeffs = np.arange(1, rt.dim + 1)
    modified = 2 * rt.tabulate(0, points)
    result = jax.jit(uflx_jax_compiler.trace(total, f, total, operand))(
        coeffs, (points, weights), modified
    )
    np.testing.assert_allclose(
        result,
        np.einsum("q,d,qdc->c", weights, coeffs, modified[0] + rt.tabulate(0, points / 2)[0]),
        atol=1e-14,
    )


def test_explicit_basis_and_fixed_tables(monkeypatch):
    interval = cell(basix.CellType.interval)
    element = basix.create_element(basix.ElementFamily.Hermite, interval.cell_type, 3)
    ref = integral(gradient(BasixBasis(function_space(interval, element), 0), interval), interval)
    (operand,) = basis_tables(ref)
    points, weights = basix.make_quadrature(interval.cell_type, 4)
    tabulate = type(element).tabulate
    values = tabulate(element, 1, points)
    calls = []

    def counted(self, order, coords):
        calls.append(order)
        return tabulate(self, order, coords)

    monkeypatch.setattr(type(element), "tabulate", counted)
    kernel = uflx_jax_compiler.trace(ref, ref, operand)
    evaluate = jax.jit(lambda weights: kernel((points, weights), values))
    for w in (weights, weights[::-1] * 2):
        np.testing.assert_allclose(evaluate(w)[:, 0], w @ values[1, :, :, 0], atol=1e-13)
    assert calls == []
    with pytest.raises(ValueError, match="not requested"):
        uflx_jax_compiler.trace(integral(num(1), interval), operand, integral(num(1), interval))


def test_table_for_a_point_integral_uses_its_declared_rule():
    point = cell(basix.CellType.point)
    element = basix.create_element(basix.ElementFamily.P, point.cell_type, 0)
    expr = integral(BasixBasis(function_space(point, element), 0), point)
    (table,) = basis_tables(expr)
    with pytest.raises(ValueError, match="explicit quadrature rule"):
        uflx_jax_compiler.trace(expr, table)
    points = np.empty((1, 0))
    result = uflx_jax_compiler.trace(expr, expr, table)(
        (points, np.ones(1)), 2 * element.tabulate(0, points)
    )
    np.testing.assert_allclose(result, [2.0])


def test_symbolic_rule_uses_caller_arrays_and_infers_point_count():
    triangle = cell(basix.CellType.triangle)
    element = basix.create_element(basix.ElementFamily.P, triangle.cell_type, 2)
    space = function_space(triangle, element)
    u, v = (function(space, label) for label in ("u", "v"))
    reference = integral(gradient(u, triangle) @ gradient(v, triangle), triangle)
    count = runtime_extent("n")
    points = runtime_array("points", (count, 2))
    weights = runtime_array("weights", (count,))
    a = form(apply_quadrature(reference, points, weights), (v, u))
    (table,) = basis_tables(a)
    assert table.point_source == points and table.order == 1
    kernel = jax.jit(uflx_jax_compiler.trace(a, points, weights, table))
    rng = np.random.default_rng(42)
    for n in (3, 5, 3, 0):
        data, w = rng.uniform(0.0, 0.45, size=(n, 2)), rng.uniform(-1, 1, size=n)
        values = rng.normal(size=(3, n, element.dim, 1))
        expected = np.einsum("q,dqi,dqj->ij", w, values[1:, :, :, 0], values[1:, :, :, 0])
        np.testing.assert_allclose(kernel(data, w, values), expected, atol=1e-13)
    with pytest.raises(ValueError, match="inconsistent array extent"):
        kernel(np.ones((3, 2)), np.ones(4), np.zeros((3, 3, element.dim, 1)))
    with pytest.raises(ValueError, match="shape"):
        apply_quadrature(
            reference, points, runtime_array("other weights", (runtime_extent("other count"),))
        )
    with pytest.raises(ValueError, match="domain-independent"):
        apply_quadrature(reference, coordinate(triangle), weights)


def test_symbolic_quadrature_mixed_fixed_operands_and_vector_integrands():
    interval = cell(basix.CellType.interval)
    element = basix.create_element(basix.ElementFamily.P, interval.cell_type, 1)
    vec = BasixBasis(function_space(interval, element), 0)
    expr = integral(vec, interval)
    points, weights = basix.make_quadrature(interval.cell_type, 2)
    p = runtime_array("p", points.shape)
    w = runtime_array("w", weights.shape)
    values = 3 * element.tabulate(0, points)
    for points_operand, weights_operand, key, data in (
        (p, weights, p, points),
        (points, w, w, weights),
    ):
        g = apply_quadrature(expr, points_operand, weights_operand)
        (table,) = basis_tables(g)
        np.testing.assert_allclose(
            jax.jit(uflx_jax_compiler.trace(g, key, table))(data, values),
            weights @ values[0, :, :, 0],
        )

    combined = apply_quadrature(expr, p, w) + apply_quadrature(expr, p, w + w)
    (shared_table,) = basis_tables(combined)
    kernel = jax.jit(uflx_jax_compiler.trace(combined, p, w, shared_table))
    np.testing.assert_allclose(kernel(points, weights, values), 3 * weights @ values[0, :, :, 0])
