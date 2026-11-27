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
from uflx.arithmetic import conjugate, num
from uflx.calculus import coordinate, gradient, integral
from uflx.forms import form, function
from uflx.geometry import domain_map, pull_back
from uflx.tensors import index, index_sum, tensor
from uflx_basix.cells import cell, create_topology
from uflx_basix.quadrature import sample_form, weighted_sum
from uflx_basix.spaces import function_space
from uflx_extensions.runtime_arrays import RuntimeArray, runtime_array, runtime_extent


def test_local_samples_share_native_values_and_mixed_partials():
    K = cell(basix.CellType.triangle)
    p2 = basix.create_element(basix.ElementFamily.P, K.cell_type, 2)
    V = function_space(K, p2)
    u, v = (function(V, name) for name in ("u", "v"))
    hessian = gradient(gradient(u, K), K)
    a = form(integral(v * (u + hessian[0, 1]) * coordinate(K)[0], K), (v, u))
    n, q = runtime_extent("n"), index("q")
    points, weights = runtime_array("points", (n, 2)), runtime_array("weights", (n,))
    samples, tables = sample_form(a, points, q)
    assert set(tables) == {(p2, 2)}
    basis = tables[p2, 2]
    assert isinstance(basis, RuntimeArray) and basis.value_shape == (6, n, p2.dim, 1)
    assert samples.value_shape == (p2.dim, p2.dim) and samples.free_indices == ((q, n),)
    evaluate = jax.jit(uflx_jax_compiler.trace(samples, points, basis))
    assemble = jax.jit(
        uflx_jax_compiler.trace(weighted_sum(samples, weights, q), points, weights, basis)
    )

    rng = np.random.default_rng(2026)
    for count in (0, 1, 4):
        pts = rng.uniform(0, 0.5, (count, 2))
        w = rng.normal(size=count)
        rows = rng.normal(size=(6, count, p2.dim, 1))
        values = rows[0, :, :, 0]
        trial = values + rows[basix.index(1, 1), :, :, 0]
        expected = np.einsum("q,qi,qj->qij", pts[:, 0], values, trial)
        np.testing.assert_allclose(evaluate(pts, rows), expected, atol=1e-13)
        np.testing.assert_allclose(
            assemble(pts, w, rows), np.einsum("q,qij->ij", w, expected), atol=1e-13
        )


def test_product_slot_packs_block_and_vector_elements_and_preserves_value_axes():
    K = cell(basix.CellType.triangle)
    p1 = basix.create_element(basix.ElementFamily.P, K.cell_type, 1)
    rt = basix.create_element(basix.ElementFamily.RT, K.cell_type, 1)
    v = function(function_space(K, p1, shape=(2,), scalar_type="complex"), "v")
    w = function(function_space(K, rt, scalar_type="complex"), "w")
    a = form(integral(num(1j) * (v + w), K), ((conjugate(v), w),))
    q = index("q")
    points, _ = basix.make_quadrature(K.cell_type, 2)
    samples, tables = sample_form(a, points, q)
    assert set(tables) == {(p1, 0), (rt, 0)}
    kernel = uflx_jax_compiler.trace(samples, tables[p1, 0], tables[rt, 0])
    scalar, vector = p1.tabulate(0, points), rt.tabulate(0, points)
    blocks = np.einsum("qi,ab->qiab", scalar[0, :, :, 0], np.eye(2))
    expected = 1j * np.concatenate((blocks.reshape(len(points), 2 * p1.dim, 2), vector[0]), axis=1)
    np.testing.assert_allclose(jax.jit(kernel)(scalar, vector), expected, atol=1e-13)


def test_sample_topology_slot_and_unused_argument():
    T = create_topology(basix.CellType.interval, basix.CellType.interval)
    K = T.cell_types[1]
    p1, p2 = (basix.create_element(basix.ElementFamily.P, K.cell_type, d) for d in (1, 2))
    v = function(function_space(T, (p1, p2)), "v")
    unused = function(function_space(K, p1), "unused")
    a = form(integral(pull_back(v, T[1]), K), (unused, v))
    pts = np.array([[0.2], [0.6]])
    samples, tables = sample_form(a, pts, index("q"))
    assert set(tables) == {(p2, 0)}
    rows = p2.tabulate(0, pts)
    result = uflx_jax_compiler.trace(samples, tables[p2, 0])(rows)
    expected = np.zeros((len(pts), p1.dim, p1.dim + p2.dim))
    expected[:, :, p1.dim :] = rows[0, :, None, :, 0]
    np.testing.assert_allclose(result, expected)


def test_sampling_boundary_and_distinct_point_operands():
    K = cell(basix.CellType.interval)
    p1 = basix.create_element(basix.ElementFamily.P, K.cell_type, 1)
    u = function(function_space(K, p1), "u")
    q, n = index("q"), runtime_extent("n")
    pts = runtime_array("points", (n, 1))
    load = integral(u, K)
    with pytest.raises(TypeError, match="Form"):
        sample_form(load, pts, q)
    shifted = pull_back(u, domain_map(K, coordinate(K) * num(0.5), K))
    with pytest.raises(ValueError, match="sampling point"):
        sample_form(form(integral(u + shifted, K), (u,)), pts, q)
    a = form(load, (u,))
    tables = sample_form(a, pts, q)[1]
    assert tables == sample_form(a, pts, index("s"))[1]
    assert tables != sample_form(a, runtime_array("other points", (n, 1)), q)[1]


def test_sampling_index_is_free_across_an_integrand_binder():
    K = cell(basix.CellType.interval)
    p1 = basix.create_element(basix.ElementFamily.P, K.cell_type, 1)
    v = function(function_space(K, p1), "v")
    q = index("q")
    a = form(integral(index_sum(tensor((v, v))[q], (q,)), K), (v,))
    points = np.array([[0.1], [0.4], [0.7]])
    samples, tables = sample_form(a, points, q)
    rows = p1.tabulate(0, points)
    assert samples.free_indices == ((q, len(points)),)
    np.testing.assert_allclose(
        uflx_jax_compiler.trace(samples, tables[p1, 0])(rows), 2 * rows[0, :, :, 0]
    )


def test_empty_reference_gradient_uses_no_table():
    K = cell(basix.CellType.point)
    p0 = basix.create_element(basix.ElementFamily.P, K.cell_type, 0)
    v = function(function_space(K, p0), "v")
    samples, tables = sample_form(
        form(integral(gradient(v, K), K), (v,)), np.empty((3, 0)), index("q")
    )
    assert tables == {}
    assert uflx_jax_compiler.trace(samples)().shape == (3, 1, 0)
