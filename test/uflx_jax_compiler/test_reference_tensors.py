# Copyright (C) 2026 Jack S. Hale
#
# This file is part of uflx-core-experimental (https://fenics.github.io/uflx/language/)
#
# SPDX-License-Identifier:    MIT
"""Reference contractions through the public form tracing interface."""

import basix
import jax
import numpy as np
import pytest

import uflx_jax_compiler
from uflx.arithmetic import logarithm, num
from uflx.calculus import gradient, integral
from uflx.forms import form, function
from uflx.geometry import push_forward
from uflx.nodes import EuclideanSpace, Mesh
from uflx.passes import lower
from uflx_basix.cells import cell, topology
from uflx_basix.quadrature import apply_quadrature
from uflx_basix.spaces import local_space
from uflx_jax_compiler import _finite_elements, _reference, reference_trace


def _constant_shapes(kernel, *data):
    return [np.shape(c) for c in jax.make_jaxpr(kernel)(*data).consts]


def test_shared_reference_form_is_computed_once():
    K = cell(basix.CellType.interval)
    p1 = basix.create_element(basix.ElementFamily.P, K.cell_type, 1)
    u = function(local_space(K, p1), "u")
    sampled = apply_quadrature(integral(u, K), *basix.make_quadrature(K.cell_type, 2))
    load = form(sampled, (u,))
    expr = load
    for _ in range(6):
        expr = expr + expr
    kernel = uflx_jax_compiler.trace(expr)
    single = jax.make_jaxpr(uflx_jax_compiler.trace(load))().jaxpr.eqns
    combined = jax.make_jaxpr(kernel)().jaxpr.eqns
    assert len(combined) == len(single) + 6
    np.testing.assert_allclose(jax.jit(kernel)(), [32.0, 32.0])


@pytest.mark.parametrize("cell_type", [basix.CellType.triangle, basix.CellType.tetrahedron])
@pytest.mark.parametrize("symmetric", [True, False], ids=["poisson", "mixed-partial"])
@pytest.mark.parametrize("degree", [1, 2])
def test_affine_reference_matrices(cell_type, symmetric, degree):
    T = topology(cell_type)
    dim = T.dimension
    p1 = basix.create_element(basix.ElementFamily.P, cell_type, 1)
    element = basix.create_element(basix.ElementFamily.P, cell_type, degree)
    x = function(local_space(T, p1, shape=(dim,)), "x")
    mesh = Mesh(x, EuclideanSpace(dim))
    V = local_space(T, element)
    v, u = function(V, "v"), function(V, "u")
    gv, gu = (gradient(push_forward(f, mesh, V.law), mesh) for f in (v, u))
    a = lower(integral(gv @ gu if symmetric else gv[0] * gu[1], mesh))
    rule = basix.make_quadrature(cell_type, 2)
    kernel = uflx_jax_compiler.trace(form(apply_quadrature(a, *rule), (v, u)), x)
    ordinary = reference_trace(form(a, (v, u)), x, a)

    coord = p1.points.copy()
    coord[:, 0] += 0.15 * coord[:, 1]
    coords = np.stack([coord, coord[::-1]])
    # The first call is inside jit/vmap. Only concrete reference arrays may be cached.
    with jax.checking_leaks():
        result = jax.jit(jax.vmap(kernel))(coords)
    reference = jax.vmap(lambda t: ordinary(t, rule))
    np.testing.assert_allclose(result, reference(coords), atol=2e-13)
    direction = np.cos(coords)
    tangent = jax.jvp(jax.vmap(kernel), (coords,), (direction,))[1]
    expected = jax.jvp(reference, (coords,), (direction,))[1]
    np.testing.assert_allclose(tangent, expected, atol=2e-13)
    n = dim * (dim + 1) // 2 if symmetric else dim**2
    has_dense_table = (n, element.dim, element.dim) in _constant_shapes(kernel, coord)
    assert has_dense_table == (degree == 2 or dim == 2)


@pytest.mark.parametrize("degree", [0, 1])
def test_nonlinear_coefficient_with_two_rules(degree):
    K = cell(basix.CellType.triangle)
    P = basix.ElementFamily.P
    E = basix.create_element(P, K.cell_type, degree, discontinuous=degree == 0)
    p2 = basix.create_element(P, K.cell_type, 2)
    c = function(local_space(K, E), "c")
    v, u = [function(local_space(K, p2), name) for name in ("v", "u")]
    mass = integral(v * u * logarithm(num(1) + c * c), K)
    weight = integral(v * u, K)
    rule = (np.array([[0.12, 0.23], [0.41, 0.17]]), np.array([0.31, 0.29]))
    user_rule = basix.make_quadrature(K.cell_type, 4)
    a = apply_quadrature(mass, *rule)
    combined = uflx_jax_compiler.trace(form(a + weight, (v, u)), c, weight)
    ordinary = uflx_jax_compiler.trace(form(mass + weight, (v, u)), c, mass, weight)
    dofs = np.linspace(0.5, 1.3, E.dim)
    np.testing.assert_allclose(
        jax.jit(combined)(dofs, user_rule),
        ordinary(dofs, rule, user_rule),
        atol=1e-13,
    )
    shapes = _constant_shapes(combined, dofs, user_rule)
    assert ((1, p2.dim, p2.dim) in shapes) == (degree == 0)

    if degree == 0:
        # The very same kernel can be given a field function. Its point dependence
        # is determined by the caller, even in a space with constant basis functions.
        field = lambda p: 0.5 + p[0]
        np.testing.assert_allclose(
            combined(field, user_rule),
            ordinary(field, rule, user_rule),
            atol=1e-13,
        )


def test_reference_table_budget_and_precision(monkeypatch):
    K = cell(basix.CellType.triangle)
    p2 = basix.create_element(basix.ElementFamily.P, K.cell_type, 2)
    v = function(local_space(K, p2), "v")
    rule = basix.make_quadrature(K.cell_type, 2)
    integral_v = integral(v, K)
    L = form(apply_quadrature(integral_v, *rule), (v,))
    kernel = uflx_jax_compiler.trace(L)
    ordinary = uflx_jax_compiler.trace(form(integral_v, (v,)), integral_v)

    def traced_basis(*args):
        pytest.fail("Fixed scalar basis values should be tabulated by Basix at quadrature points")

    for x64 in (False, True):
        jax.config.update("jax_enable_x64", x64)
        expected = ordinary(rule)
        with monkeypatch.context() as patch:
            patch.setattr(_finite_elements, "tabulate", traced_basis)
            result = kernel()
        assert result.dtype == expected.dtype == (np.float64 if x64 else np.float32)
        np.testing.assert_allclose(result, expected, atol=1e-7 if not x64 else 1e-14)

    monkeypatch.setattr(_reference, "_MAX_ENTRIES", 1)
    limited = uflx_jax_compiler.trace(L)
    assert (1, p2.dim) not in _constant_shapes(limited)
    np.testing.assert_allclose(limited(), ordinary(rule), atol=1e-14)


def test_explicit_point_measure_in_a_sampled_form():
    K = cell(basix.CellType.triangle)
    p2 = basix.create_element(basix.ElementFamily.P, K.cell_type, 2)
    v = function(local_space(K, p2), "v")
    rule = basix.make_quadrature(K.cell_type, 2)
    sampled = apply_quadrature(integral(v, K), *rule)
    point_integral = sampled.expression.second
    L = form(sampled, (v,))
    reweighted = uflx_jax_compiler.trace(L, (point_integral, point_integral.integration_domain))
    actual = reweighted((np.empty((2, 0)), [0.2, 0.3]))
    np.testing.assert_allclose(actual, 0.5 * uflx_jax_compiler.trace(L)(), atol=1e-14)
