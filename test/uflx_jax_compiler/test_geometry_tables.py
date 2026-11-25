# Copyright (C) 2026 Jack S. Hale
#
# This file is part of uflx-core-experimental (https://fenics.github.io/uflx/language/)
#
# SPDX-License-Identifier:    MIT

import basix
import jax
import numpy as np
import pytest

import uflx_jax_compiler
from uflx.algorithms import substitute_values
from uflx.arithmetic import conditional, less_than, num
from uflx.calculus import coordinate, gradient, integral
from uflx.forms import form, function
from uflx.geometry import create_mesh, density, euclidean_space, jacobian, push_forward
from uflx.passes import lower
from uflx.tensors import index, inverse
from uflx_basix.cells import cell, topology
from uflx_basix.nodes import table
from uflx_basix.quadrature import apply_quadrature, evaluate_at_points, weighted_sum
from uflx_basix.spaces import local_space
from uflx_extensions.runtime_arrays import runtime_array, runtime_extent
from uflx_jax_compiler import reference_trace


def test_sample_once_and_reduce_with_independent_weights():
    triangle = cell(basix.CellType.triangle)
    q = index("q")
    n = runtime_extent("n")
    points = runtime_array("points", (n, triangle.dimension))
    weights = runtime_array("weights", (n,))
    position = coordinate(triangle)
    samples = evaluate_at_points(position[0] * position, points, q)
    sample = jax.jit(uflx_jax_compiler.trace(samples, points))
    values = runtime_array("values", (n, *position.value_shape))
    reduce_samples = jax.jit(
        uflx_jax_compiler.trace(weighted_sum(values[q], weights, q), values, weights)
    )
    fused = jax.jit(uflx_jax_compiler.trace(weighted_sum(samples, weights, q), points, weights))
    for count in (0, 1, 3):
        pts = np.array([[0.2, 0.1], [0.1, 0.6], [0.5, 0.3]])[:count]
        rows = sample(pts)
        np.testing.assert_allclose(rows, pts[:, :1] * pts)
        for w in (np.ones(count), np.arange(count, dtype=float)):
            expected = w @ (pts[:, :1] * pts)
            np.testing.assert_allclose(reduce_samples(rows, w), expected)
            np.testing.assert_allclose(fused(pts, w), expected)


def test_mass_and_stiffness_extract_and_share_curved_geometry():
    triangle = basix.CellType.triangle
    T = topology(triangle)
    p2 = basix.create_element(basix.ElementFamily.P, triangle, 2)
    V = local_space(T, p2)
    u, v, k = (function(V, name) for name in ("u", "v", "conductivity"))
    x = function(local_space(T, p2, shape=(2,)), "x")
    mesh = create_mesh(x, euclidean_space(2))
    trial, test, conductivity = (push_forward(f, mesh, V.law) for f in (u, v, k))
    stiffness = conductivity * (gradient(test, mesh) @ gradient(trial, mesh))
    mass = conductivity * test * trial
    n = runtime_extent("n")
    q = index("q")
    points = runtime_array("points", (n, 2))
    weights = runtime_array("weights", (n,))

    s, t = p2.points.T
    coords = np.column_stack((s, t + 0.4 * s * t))
    coefficient = 2 + s - t
    density_term = density(mesh[0])
    definitions = {}
    for integrand, terms in (
        (mass, {density_term}),
        (stiffness, {inverse(jacobian(mesh[0])), density_term}),
    ):
        physical = integral(integrand, mesh)
        reference = lower(physical)
        residual, geometry = uflx_jax_compiler.extract_geometry(reference, mesh[0])
        assert set(geometry.values()) == terms
        for operand, term in geometry.items():
            assert definitions.setdefault(term, operand) is operand
            assert operand.value_shape == term.value_shape
        evaluators = []
        for term in geometry.values():
            sample = evaluate_at_points(term, points, q)
            (derivatives,) = uflx_jax_compiler.basis_tables(sample)
            assert derivatives.element == p2 and derivatives.order == 1
            evaluators.append(uflx_jax_compiler.trace(sample, x, points, derivatives))
        with pytest.raises(ValueError, match="reference cell"):
            uflx_jax_compiler.extract_geometry(physical, mesh[0])
        original = apply_quadrature(reference, points, weights)
        with pytest.raises(TypeError, match="Integral"):
            uflx_jax_compiler.extract_geometry(original, mesh[0])
        direct = jax.jit(reference_trace(form(original, (v, u)), x, k, points, weights))
        inputs = {f: runtime_array(f.label, (n, *f.value_shape)) for f in geometry}
        body = substitute_values(residual.expression, {f: data[q] for f, data in inputs.items()})
        samples = evaluate_at_points(body, points, q)
        a = form(weighted_sum(samples, weights, q), (v, u))
        (basis,) = uflx_jax_compiler.basis_tables(a)
        assemble = jax.jit(uflx_jax_compiler.trace(a, k, points, weights, *inputs.values(), basis))
        for degree in (2, 4):
            pts, w = basix.make_quadrature(triangle, degree)
            derivatives = p2.tabulate(1, pts)
            for vertices in (coords, coords[:, ::-1]):
                tables = [evaluate(vertices, pts, derivatives) for evaluate in evaluators]
                actual = assemble(coefficient, pts, w, *tables, p2.tabulate(basis.order, pts))
                np.testing.assert_allclose(
                    actual, direct(vertices, coefficient, pts, w), atol=1e-13
                )


def test_mixed_gradient_keeps_its_geometry_and_other_fields():
    interval = basix.CellType.interval
    T = topology(interval)
    p2 = basix.create_element(basix.ElementFamily.P, interval, 2)
    x = function(local_space(T, p2, shape=(1,)), "x")
    cell_map = create_mesh(x, euclidean_space(1))[0]
    cell = cell_map.reference_cell
    u = function(local_space(cell, p2), "u")
    stretch = jacobian(cell_map)[0, 0]
    points, weights = basix.make_quadrature(interval, 4)
    mixed = gradient(stretch * u, cell)[0]
    mixed_integral = integral(mixed, cell)
    assert uflx_jax_compiler.extract_geometry(mixed_integral, cell_map) == (mixed_integral, {})

    original = integral(mixed + stretch, cell)
    residual, geometry = uflx_jax_compiler.extract_geometry(original, cell_map)
    ((operand, term),) = geometry.items()
    assert term is stretch
    s = p2.points[:, 0]
    coords = (s + s**2)[:, None]
    q = index("q")
    values = uflx_jax_compiler.trace(evaluate_at_points(term, points, q), x)(coords)
    np.testing.assert_allclose(values, 1 + 2 * points[:, 0])
    body = substitute_values(residual.expression, {operand: table(values)[q]})
    weighted = weighted_sum(evaluate_at_points(body, points, q), weights, q)
    actual = reference_trace(weighted, u, x)(s**2, coords)
    direct = apply_quadrature(original, points, weights)
    np.testing.assert_allclose(actual, reference_trace(direct, u, x)(s**2, coords))


def test_pointwise_table_binding_preserves_conditional_evaluation():
    cell = topology(basix.CellType.interval).cell_types[0]
    value = runtime_array("value", ())
    missing = runtime_array("missing", ())
    expr = conditional(less_than(value, num(10)), value, missing)
    q = index("q")
    body = substitute_values(expr, {value: table([1, 2])[q]})
    samples = evaluate_at_points(body, [[0.25], [0.75]], q, cell=cell)
    weighted = weighted_sum(samples, [0.5, 0.5], q)
    assert uflx_jax_compiler.trace(weighted)() == 1.5
