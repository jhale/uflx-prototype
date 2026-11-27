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
from uflx.algorithms import bottom_up, substitute_values
from uflx.arithmetic import logarithm, num
from uflx.calculus import coordinate, gradient, integral
from uflx.forms import form, function
from uflx.geometry import push_forward
from uflx.nodes import EuclideanSpace, Integral, Mesh, PullBack
from uflx.passes import expand_pull_backs, lower
from uflx.tensors import index, tensor
from uflx_basix.cells import cell, topology
from uflx_basix.quadrature import apply_quadrature
from uflx_basix.spaces import function_space

from ..testutil import contains
from ..uflx.spaces import space


def test_choose_rules_for_distinct_integrals_on_the_same_cell():
    K = cell(basix.CellType.interval)
    x = coordinate(K)[0]
    a, b, c = (integral(x ** num(power), K) for power in (2, 3, 4))
    expression = tensor((a, b, c, a))
    partial = substitute_values(
        expression,
        {
            a: apply_quadrature(a, [[0.25]], [1.0]),
            b: apply_quadrature(b, *basix.make_quadrature(K.cell_type, 4)),
        },
    )
    assert partial.entries[2] is c
    assert partial.entries[0] is partial.entries[3]
    np.testing.assert_allclose(
        uflx_jax_compiler.trace(partial, c)(basix.make_quadrature(K.cell_type, 4)),
        [1 / 16, 1 / 4, 1 / 5, 1 / 16],
    )

    finished = substitute_values(
        partial, {c: apply_quadrature(c, *basix.make_quadrature(K.cell_type, 1))}
    )
    assert finished.entries[:2] == partial.entries[:2]
    np.testing.assert_allclose(uflx_jax_compiler.trace(finished)(), [1 / 16, 1 / 4, 1 / 16, 1 / 16])


def test_trace_requires_a_rule_for_each_positive_dimensional_integral():
    K = cell(basix.CellType.interval)
    x = coordinate(K)[0]
    first, second = integral(x, K), integral(x * x, K)
    moments = tensor((first, second))
    for declared in ((), (first,), (second,)):
        with pytest.raises(ValueError, match="explicit quadrature"):
            uflx_jax_compiler.trace(moments, *declared)

    kernel = uflx_jax_compiler.trace(moments, first, second)
    midpoint = ([[0.5]], [1.0])
    left = ([[0.25]], [1.0])
    np.testing.assert_allclose(kernel(midpoint, left), [0.5, 0.0625])
    np.testing.assert_allclose(kernel(left, midpoint), [0.25, 0.25])
    with pytest.raises(ValueError, match="one reference point per weight"):
        kernel(([[0.5]], [0.5, 0.5]), left)


def test_one_integral_or_a_bottom_up_traversal_of_nested_integrals():
    K = cell(basix.CellType.interval)
    x = coordinate(K)[0]
    inner = integral(x * x, K)
    outer = integral(inner * x, K)
    rule = ([[0.25]], [1.0])
    # Sampling just the outer integral leaves the inner integral equal to 1/3.
    assert uflx_jax_compiler.trace(apply_quadrature(outer, *rule), inner)(
        basix.make_quadrature(K.cell_type, 2)
    ) == pytest.approx(1 / 12)

    def sample(node):
        if isinstance(node, Integral) and node.integration_domain == K:
            return apply_quadrature(node, *rule)
        return node

    assert uflx_jax_compiler.trace(bottom_up(outer, sample))() == pytest.approx(1 / 64)


def test_point_evaluations_can_be_expanded_after_sampling():
    K = cell(basix.CellType.triangle)
    moment = integral(coordinate(K)[0], K)
    sampled = apply_quadrature(moment, *basix.make_quadrature(K.cell_type, 2))
    assert contains(sampled, lambda node: isinstance(node, PullBack))
    expanded = expand_pull_backs(sampled)
    assert not contains(expanded, lambda node: isinstance(node, PullBack))
    assert uflx_jax_compiler.trace(expanded)() == pytest.approx(1 / 6)


def test_custom_rule_preserves_tensor_and_free_axes():
    K = cell(basix.CellType.triangle)
    x = coordinate(K)
    i = index("row")
    expr = integral(tensor((num(1), num(2)))[i] * x, K)
    points, weights = basix.make_quadrature(K.cell_type, 2, rule=basix.QuadratureType.xiao_gimbutas)
    quadrature = apply_quadrature(expr, points, weights)
    assert (
        quadrature.value_shape == expr.value_shape and quadrature.free_indices == expr.free_indices
    )
    np.testing.assert_allclose(
        uflx_jax_compiler.trace(quadrature)(), [[1 / 6, 1 / 6], [1 / 3, 1 / 3]]
    )
    empty = apply_quadrature(expr, points[:0], weights[:0])
    assert empty.free_indices == expr.free_indices
    np.testing.assert_array_equal(uflx_jax_compiler.trace(empty)(), np.zeros((2, 2)))
    np.testing.assert_array_equal(
        uflx_jax_compiler.trace(expr, expr)((points[:0], weights[:0])), np.zeros((2, 2))
    )


def test_explicit_weights_on_a_point_cell():
    expr = integral(num(3), cell(basix.CellType.point))
    assert uflx_jax_compiler.trace(expr)() == 3
    result = apply_quadrature(expr, np.empty((2, 0)), [0.5, 1.5])
    assert uflx_jax_compiler.trace(result)() == 6


@pytest.mark.parametrize("cell_type", [basix.CellType.quadrilateral, basix.CellType.hexahedron])
def test_basix_rule_on_a_tensor_product_cell(cell_type):
    K = cell(cell_type)
    x = coordinate(K)
    moment = integral(x[0] * x[0] + x[K.dimension - 1], K)
    assert uflx_jax_compiler.trace(
        apply_quadrature(moment, *basix.make_quadrature(K.cell_type, 2))
    )() == pytest.approx(5 / 6)


@pytest.mark.parametrize("degree", [1, 2])
def test_coefficient_gradient_uses_only_its_dependent_axes(degree):
    K = cell(basix.CellType.triangle)
    P = basix.ElementFamily.P
    element = basix.create_element(P, K.cell_type, degree)
    linear = basix.create_element(P, K.cell_type, 1)
    u = function(function_space(K, element), "u")
    v = function(function_space(K, linear), "v")
    points, weights = basix.make_quadrature(K.cell_type, 4)
    moment = integral(v * logarithm(gradient(u, K)[0]), K)
    kernel = uflx_jax_compiler.trace(form(apply_quadrature(moment, points, weights), (v,)), u)

    x = element.points[:, 0]
    dofs = 2 * x + (degree - 1) * x**2
    slope = 2 + (degree - 1) * 2 * points[:, 0]
    expected = (weights * np.log(slope)) @ linear.tabulate(0, points)[0, :, :, 0]
    np.testing.assert_allclose(jax.jit(kernel)(dofs), expected)
    eqns = jax.make_jaxpr(kernel)(dofs).jaxpr.eqns
    assert [e.invars[0].aval.shape for e in eqns if e.primitive.name == "log"] == [
        () if degree == 1 else (len(points),)
    ]


def test_material_axis_stays_independent_of_quadrature():
    K = cell(basix.CellType.triangle)
    i = index("material")
    c = function(space("materials", None, (2,), scalar_type="real"), "c")
    moment = integral(coordinate(K)[0] * logarithm(c[i]), K)
    sampled = apply_quadrature(moment, *basix.make_quadrature(K.cell_type, 4))
    kernel = uflx_jax_compiler.trace(sampled, c)
    values = np.array([2.0, 3.0])
    np.testing.assert_allclose(jax.jit(kernel)(values), np.log(values) / 6)
    eqns = jax.make_jaxpr(kernel)(values).jaxpr.eqns
    assert [e.invars[0].aval.shape for e in eqns if e.primitive.name == "log"] == [(2,)]


def test_poisson_stiffness_on_a_curved_triangle():
    triangle = basix.CellType.triangle
    T = topology(triangle)
    P = basix.ElementFamily.P
    p1 = basix.create_element(P, triangle, 1)
    p2 = basix.create_element(P, triangle, 2)
    x = function(function_space(T, p2, shape=(2,)), "x")
    mesh = Mesh(x, EuclideanSpace(2))
    V = function_space(T, p1)
    u, v = function(V, "u"), function(V, "v")
    gu, gv = (gradient(push_forward(f, mesh, V.law), mesh) for f in (u, v))
    points, weights = basix.make_quadrature(triangle, 4)
    sampled = apply_quadrature(lower(integral(gu @ gv, mesh)), points, weights)
    stiffness = jax.jit(uflx_jax_compiler.trace(form(sampled, (v, u)), x))

    # x(s, t) = (s + s², t), so J is diagonal with entries (1 + 2s, 1).
    coordinates = p2.points.copy()
    coordinates[:, 0] += coordinates[:, 0] ** 2
    stretch = 1 + 2 * points[:, 0]
    grad = np.stack(p1.tabulate(1, points)[1:, :, :, 0], axis=-1)
    grad[:, :, 0] /= stretch[:, None]
    expected = np.einsum("q,qir,qjr->ij", weights * stretch, grad, grad)
    np.testing.assert_allclose(stiffness(coordinates), expected, atol=1e-12)
