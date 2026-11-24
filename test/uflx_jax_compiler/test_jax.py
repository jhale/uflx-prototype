# Copyright (C) 2026 Jack S. Hale
#
# This file is part of uflx-core-experimental (https://fenics.github.io/uflx/language/)
#
# SPDX-License-Identifier:    MIT
"""Trace the original UFLx graph through the public evaluation entry point."""

import math
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from uflx.arithmetic import conditional, conjugate, less_than, num
from uflx.calculus import coordinate, gradient, integral
from uflx.forms import apply, conjugate_space, dual_space, form, function
from uflx.forms import partial as partial_form
from uflx.geometry import pull_back
from uflx_basix.quadrature import apply_quadrature
from uflx_extensions.external_operators import external_operator
from uflx_jax_compiler import trace
from uflx_jax_compiler.interpreter import Pointwise

from ..uflx.spaces import space as field_space
from ..uflx_basix.cells import reference_simplex
from ..uflx_extensions.operations import Energy
from .test_differentiation import Sine


def test_trace_shared_arithmetic_once():
    u = function(field_space("scalar", None, scalar_type="real"), "u")
    expr = u
    for _ in range(12):
        expr = expr + expr
    kernel = trace(expr, u)
    eqns = jax.make_jaxpr(kernel)(1.0).jaxpr.eqns
    assert sum(eqn.primitive.name == "add" for eqn in eqns) == 12
    assert jax.jit(kernel)(3.0) == 3 * 2**12


def test_shared_expression_in_separate_conditional_traces():
    u = function(field_space("scalar", None, scalar_type="real"), "u")
    square = u * u
    piece = conditional(less_than(u, num(0)), square + u, square - u)
    kernel = trace(piece + square, u)
    with jax.checking_leaks():
        values, gradients = jax.jit(jax.vmap(jax.value_and_grad(kernel)))(jnp.array([-2.0, 3.0]))
    np.testing.assert_allclose(values, [6.0, 15.0])
    np.testing.assert_allclose(gradients, [-7.0, 11.0])


def test_shared_form_uses_distinct_factor_arguments():
    u = function(field_space("vector", None, (2,), scalar_type="real"), "u")
    linear = form(u[0] + u[1], (u,))
    kernel = trace(linear * linear)
    assert jax.jit(kernel)(jnp.array([1.0, 2.0]), jnp.array([3.0, 4.0])) == 21.0


def test_jit_value_and_gradient_of_integrated_spatial_derivative():
    cell = reference_simplex(1)
    x = coordinate(cell)[0]
    a = function(field_space("scale", None, scalar_type="real"), "a")
    slope = gradient(a * x * x, cell)[0]
    expression = integral(slope * slope, cell)

    energy = partial(trace(expression, expression, a), cell.quadrature(2))

    values, gradients = jax.jit(jax.vmap(jax.value_and_grad(energy)))(jnp.array([3.0, 1.5]))
    np.testing.assert_allclose(values, [12, 3])
    np.testing.assert_allclose(gradients, [8, 4])


def test_callable_takes_declared_inputs_then_ordered_form_slots():
    V = field_space("V", None, scalar_type="real")
    c = function(V, "c")
    W = field_space("W", None, (2,), scalar_type="real")
    u, v = function(W, "u"), function(W, "v")
    operator = form(c * u[0] * v[1], (u, v))
    kernel = trace(operator, c)
    first, second = jnp.array([1.0, 2.0]), jnp.array([3.0, 4.0])
    result = kernel(2.0, first, second)
    assert isinstance(result, jax.Array)
    assert result == 8.0
    assert jax.jit(kernel)(3.0, second, first) == 18.0
    with pytest.raises(TypeError, match="expected 3 positional values"):
        kernel(2.0, first)


def test_local_form_axes_and_applied_coefficients():
    import basix

    from uflx_basix.cells import cell
    from uflx_basix.spaces import local_space

    K = cell(basix.CellType.interval)
    V = local_space(K, basix.create_element(basix.ElementFamily.P, K.cell_type, 1))
    W = local_space(K, basix.create_element(basix.ElementFamily.P, K.cell_type, 2))
    v, u = function(V, "v"), function(W, "u")
    rule = basix.make_quadrature(K.cell_type, degree=3)
    L = form(apply_quadrature(integral(u, K), *rule), (u,))
    mass = form(apply_quadrature(integral(v * u, K), *rule), (v, u))

    b = jax.jit(trace(L))()
    A = jax.jit(trace(mass))()
    np.testing.assert_allclose(b, [1 / 6, 1 / 6, 2 / 3])
    np.testing.assert_allclose(A, [[1 / 6, 0, 1 / 3], [0, 1 / 6, 1 / 3]], atol=1e-13)

    u_dofs = jnp.array([-1.0, 3.0, 5.0])
    np.testing.assert_allclose(trace(apply(L, (u,)), u)(u_dofs), b @ u_dofs)
    w = function(W, "w")
    np.testing.assert_allclose(trace(partial_form(mass, {u: w}), w)(u_dofs), A @ u_dofs)
    v_dofs = jnp.array([2.0, 4.0])
    action = trace(apply(mass, (v, u)), v, u)
    np.testing.assert_allclose(action(v_dofs, u_dofs), v_dofs @ A @ u_dofs)
    with pytest.raises(TypeError, match="expected 0 positional values"):
        trace(L)(u_dofs)

    # A supplied functional receives each basis field and chooses where to sample it.
    ell = function(dual_space(V), "ell")
    np.testing.assert_allclose(trace(ell, ell)(lambda f: f(jnp.array([0.25]))), [0.75, 0.25])


def test_product_basis_with_blocked_and_conjugate_fields():
    import basix

    from uflx_basix.cells import cell
    from uflx_basix.spaces import local_space

    K = cell(basix.CellType.interval)
    p1 = basix.create_element(basix.ElementFamily.P, K.cell_type, 1)
    W = local_space(K, p1, shape=(2,), scalar_type="complex")
    V = local_space(K, p1, scalar_type="complex")
    v, q = function(W, "v"), function(conjugate_space(V), "q")
    c = function(field_space("scalar", None, scalar_type="real"), "c")
    field = conjugate(v[0]) + num(2) * conjugate(v[1]) + num(3) * q
    rule = basix.make_quadrature(K.cell_type, degree=1)
    mixed = form(apply_quadrature(integral(c * field, K), *rule), (c, (conjugate(v), q)))

    b = jax.jit(trace(mixed))(4.0)
    assert jnp.iscomplexobj(b)
    np.testing.assert_allclose(b, [2, 4, 2, 4, 6, 6])


def test_jit_differentiates_whole_field_samples():
    cell = reference_simplex(1)
    a = function(field_space("scale", None, scalar_type="real"), "a")
    x = coordinate(cell)[0]
    law = Energy()
    expression = external_operator(law, (a * x, a))

    def read(field, scale):
        return scale * (field(jnp.array([0.25])) + field(jnp.array([0.75])))

    run = partial(trace(expression, law, a), read)

    val, deriv = jax.jit(jax.value_and_grad(run))(2.0)
    np.testing.assert_allclose([val, deriv], [4.0, 4.0])


def test_basix_tabulation_at_traced_points_and_coefficients():
    import basix

    from uflx_basix.cells import cell
    from uflx_basix.spaces import local_space

    K = cell(basix.CellType.interval)
    V = local_space(K, basix.create_element(basix.ElementFamily.P, K.cell_type, 2))
    u = function(V, "u")
    dofs = V.element(0).points[:, 0] ** 2
    slope = gradient(u, K)[0]

    sample = trace(slope, u, K)

    f = jax.jit(jax.vmap(jax.value_and_grad(sample, argnums=1), in_axes=(None, 0)))
    values, hessians = f(dofs, jnp.array([[0.375], [0.75]]))
    np.testing.assert_allclose(values, [0.75, 1.5])
    np.testing.assert_allclose(hessians, [[2.0], [2.0]])


def test_pullback_selects_the_cell_of_a_discontinuous_field():
    import basix

    from uflx_basix.cells import topology
    from uflx_basix.spaces import local_space

    triangle = basix.CellType.triangle
    T = topology(triangle, triangle)
    p0 = basix.create_element(basix.ElementFamily.P, triangle, 0, discontinuous=True)
    u = function(local_space(T, p0), "u")
    dofs = jnp.array([2.0, 7.0])
    point = jnp.array([0.0, 0.0])

    for slot, piece in enumerate(T):
        restriction = pull_back(u, piece)
        sample = trace(restriction, u, piece.reference_cell)
        assert jax.jit(sample)(dofs, point) == pytest.approx(dofs[slot])
        basis = trace(form(restriction, (u,)), piece.reference_cell)
        np.testing.assert_allclose(basis(point), np.eye(2)[slot])


def test_batched_basix_mixed_derivatives_on_a_tetrahedron():
    import basix

    from uflx_basix.cells import cell
    from uflx_basix.spaces import local_space

    K = cell(basix.CellType.tetrahedron)
    V = local_space(
        K,
        basix.create_element(
            basix.ElementFamily.P, K.cell_type, 3, basix.LagrangeVariant.equispaced
        ),
    )
    u = function(V, "u")
    r, s, t = V.element(0).points.T
    dr_ds = gradient(gradient(u, K), K)[0, 1]

    sample = partial(trace(dr_ds, u, K), r * s * t + r * r * s)

    points = jnp.array([[0.2, 0.3, 0.1], [0.4, 0.1, 0.25]])
    mixed, derivatives = jax.jit(jax.vmap(jax.value_and_grad(sample)))(points)
    np.testing.assert_allclose(mixed, 2 * points[:, 0] + points[:, 2], atol=1e-12)
    np.testing.assert_allclose(derivatives, [[2, 0, 1], [2, 0, 1]], atol=1e-12)


@pytest.mark.parametrize("reference_precision", [False], indirect=True)
def test_external_jax_custom_jvp_transposes_under_jit(reference_precision):
    a = function(field_space("scale", None, scalar_type="real"), "a")
    law = Energy()
    expr = external_operator(law, (a, -a))

    @jax.custom_jvp
    def maximum(x, y):
        return jnp.maximum(x, y)

    @maximum.defjvp
    def tangent(primals, directions):
        x, y = primals
        dx, dy = directions
        # Choose the first operand's derivative at a tie.
        return maximum(x, y), jnp.where(x >= y, dx, dy)

    read = partial(trace(expr, law, a), maximum)

    assert jax.jit(jax.grad(read))(0.0) == 1.0


def test_spatial_derivative_of_a_field_returned_by_a_library():
    cell = reference_simplex(1)
    a = function(field_space("scale", None, scalar_type="real"), "a")
    x = coordinate(cell)[0]
    law = Sine()
    expr = external_operator(law, (a * x,))

    def operation(field):
        return lambda p: p[0] * jnp.sin(field(p))

    evaluate = partial(trace(expr, law, a, cell), operation)

    value, (da, dx) = jax.jit(jax.value_and_grad(evaluate, argnums=(0, 1)))(2.0, jnp.array([3.0]))
    np.testing.assert_allclose(
        [value, da, dx[0]], [3 * math.sin(6), 9 * math.cos(6), math.sin(6) + 6 * math.cos(6)]
    )


def test_pointwise_jax_binding_is_traced_directly():
    a = function(field_space("scale", None, scalar_type="real"), "a")
    law = Sine()
    expr = external_operator(law, (a,))

    read = partial(trace(expr, law, a), Pointwise(jnp.sin))

    np.testing.assert_allclose(jax.jit(jax.grad(read))(0.5), math.cos(0.5))


@pytest.mark.parametrize(
    ("cell_name", "family", "degree"),
    [
        ("triangle", "P", 4),
        ("tetrahedron", "RT", 2),
        ("quadrilateral", "P", 3),
        ("prism", "P", 2),
        ("pyramid", "P", 2),
        ("triangle", "iso", 2),
        ("tetrahedron", "iso", 1),
    ],
)
def test_native_basis_values_and_derivatives(cell_name, family, degree):
    import basix

    from uflx_basix.cells import cell
    from uflx_basix.nodes import BasixBasis
    from uflx_basix.spaces import local_space

    K = cell(basix.CellType[cell_name])
    element = basix.create_element(
        basix.ElementFamily[family], K.cell_type, degree, basix.LagrangeVariant.equispaced
    )
    basis = trace(BasixBasis(local_space(K, element), 0), K)
    points, _ = basix.make_quadrature(K.cell_type, 4)
    points = np.vstack((points, basix.geometry(K.cell_type)))
    sample = jax.vmap(lambda p: (basis(p), jax.jacfwd(basis)(p)))
    assert "callback" not in str(jax.make_jaxpr(sample)(points))
    values, gradients = jax.jit(sample)(points)
    native = element.tabulate(1, points)
    np.testing.assert_allclose(values, native[0].reshape(values.shape), atol=1e-12)
    for axis in range(K.dimension):
        row = basix.index(*np.eye(K.dimension, dtype=int)[axis])
        np.testing.assert_allclose(
            gradients[..., axis], native[row].reshape(values.shape), atol=1e-11
        )


@pytest.mark.parametrize("cell_name", ["interval", "quadrilateral", "hexahedron"])
def test_macroedge_uses_the_polynomial_on_the_selected_half(cell_name):
    import basix

    from uflx_basix.cells import cell
    from uflx_basix.spaces import local_space

    K = cell(basix.CellType[cell_name])
    element = basix.create_element(
        basix.ElementFamily.iso, K.cell_type, 1, basix.LagrangeVariant.equispaced
    )
    u = function(local_space(K, element), "u")
    # This continuous piecewise linear field has a kink at the midpoint.
    dofs = np.maximum(element.points[:, 0] - 0.5, 0.0)
    kernel = partial(trace(u, u, K), dofs)
    points = np.tile(np.array([[0.25], [0.5], [0.75]]), (1, K.dimension))
    values, gradients = jax.jit(jax.vmap(jax.value_and_grad(kernel)))(points)
    np.testing.assert_allclose(values, [0, 0, 0.25], atol=1e-13)
    expected = np.zeros_like(points)
    expected[2, 0] = 1
    np.testing.assert_allclose(gradients, expected, atol=1e-13)
