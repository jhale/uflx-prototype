# Copyright (C) 2026 Jack S. Hale
#
# This file is part of uflx-core-experimental (https://fenics.github.io/uflx/language/)
#
# SPDX-License-Identifier:    MIT
"""Free-index axes, tensor value axes and the boundaries between them."""

import jax
import jax.numpy as jnp
import numpy as np

import uflx_jax_compiler
from uflx.arithmetic import conditional, less_than, num
from uflx.calculus import coordinate, gradient
from uflx.forms import form, function
from uflx.geometry import domain_map, pull_back
from uflx.nodes import EuclideanSpace, GatheredIndex
from uflx.tensors import broadcast_to, component_tensor, index, index_sum, tensor
from uflx_extensions.runtime_arrays import RuntimeArray, RuntimeExtent
from uflx_jax_compiler import reference_trace

from ..uflx.library import IndexTable
from ..uflx.spaces import space


def test_contraction_retains_free_axes_in_order_and_differentiates():
    i, j, k = map(index, "ijk")
    A, B = RuntimeArray("A", (2, 3)), RuntimeArray("B", (4, 3))
    b = RuntimeArray("b", (3,))
    expr = index_sum(A[i, j] * B[k, j] * b[j], (j,))
    a, m, v = jnp.arange(6.0).reshape(2, 3), jnp.arange(12.0).reshape(4, 3), jnp.arange(3.0)
    expected = jnp.einsum("ij,kj,j->ik", a, m, v)
    np.testing.assert_allclose(uflx_jax_compiler.trace(expr, A, B, b)(a, m, v), expected)
    np.testing.assert_allclose(
        uflx_jax_compiler.trace(expr, A, B, b, k)(a, m, v, 2), expected[:, 2]
    )
    # Rebinding i as a tensor axis shadows the caller's selection.
    np.testing.assert_allclose(
        uflx_jax_compiler.trace(component_tensor(expr, (i,)), A, B, b, i)(a, m, v, 1), expected.T
    )

    def evaluate(v):
        return uflx_jax_compiler.trace(expr, A, B, b)(a, m, v)

    np.testing.assert_allclose(jax.jit(jax.jacrev(evaluate))(v), jnp.einsum("ij,kj->ikj", a, m))


def test_value_axes_follow_free_axes_during_broadcasts_and_gathers():
    i, j = map(index, "ij")
    a, b = RuntimeArray("a", (2,)), RuntimeArray("b", (3,))
    first, second = [1.0, 2.0], [3.0, 4.0, 5.0]
    pair = tensor((a[i], b[j]))
    scaled = b[j] * broadcast_to(pair, (4, 2))
    expected = np.empty((3, 2, 4, 2))
    for y, x in np.ndindex(3, 2):
        expected[y, x] = (y + 3) * np.array([x + 1, y + 3])
    np.testing.assert_array_equal(uflx_jax_compiler.trace(scaled, a, b)(first, second), expected)
    reversed_i = GatheredIndex(IndexTable((1, 0), 2), (i,))
    np.testing.assert_array_equal(
        uflx_jax_compiler.trace(pair[reversed_i], a, b)(first, second), [[3, 4, 5], [2, 2, 2]]
    )


def test_empty_runtime_contraction_does_not_read_the_body():
    i, j = map(index, "ij")
    n = RuntimeExtent("n")
    a = RuntimeArray("a", (3, n))
    expr = a[i, j] * a[i, j]
    np.testing.assert_array_equal(uflx_jax_compiler.trace(index_sum(expr, (j,)), n)(0), np.zeros(3))
    assert uflx_jax_compiler.trace(component_tensor(expr, (j, i)), n)(0).shape == (0, 3)


def test_family_of_pullbacks_and_lazy_conditionals():
    i = index("i")
    line = EuclideanSpace(1)
    a = RuntimeArray("a", (3,))
    points = domain_map(EuclideanSpace(0), tensor((a[i],)), line)
    x = coordinate(line)[0]
    expr = pull_back(gradient(x * x, line), points)
    values = [-1.0, 2.0, 3.0]
    np.testing.assert_allclose(reference_trace(expr, a)(values), [[-2.0], [4.0], [6.0]])
    missing = RuntimeArray("unused", ())
    selected = conditional(less_than(a[i], num(10)), a[i], missing)
    np.testing.assert_array_equal(uflx_jax_compiler.trace(selected, a)(values), [-1.0, 2.0, 3.0])
    field = conditional(less_than(x, num(10)), x, missing)
    np.testing.assert_array_equal(
        uflx_jax_compiler.trace(pull_back(field, points), a)(values), [-1.0, 2.0, 3.0]
    )
    u = function(space("field", line, scalar_type="real"), "u")
    np.testing.assert_array_equal(
        uflx_jax_compiler.trace(pull_back(u, points), a, u)(values, field), [-1.0, 2.0, 3.0]
    )


def test_indexed_pullback_preserves_shared_axes_under_jit_and_ad():
    i, j, k = map(index, "ijk")
    line = EuclideanSpace(1)
    a, b = RuntimeArray("a", (2, 3)), RuntimeArray("b", (4, 3))
    points = domain_map(EuclideanSpace(0), tensor((a[i, j],)), line)
    x = coordinate(line)[0]
    field = tensor((b[k, j] * x, b[k, j] + x))
    expr = pull_back(field, points)
    locations, weights = jnp.arange(6.0).reshape(2, 3), jnp.arange(12.0).reshape(4, 3)

    def sample(locations):
        return uflx_jax_compiler.trace(expr, a, b)(locations, weights)

    # Field axes (k, j), then the additional map axis i, then the two components.
    w, p = weights[:, :, None], locations.T[None, :, :]
    expected = jnp.stack((w * p, w + p), axis=-1)
    np.testing.assert_allclose(jax.jit(sample)(locations), expected)
    derivative = jax.grad(lambda a: sample(a).sum())(locations)
    np.testing.assert_allclose(derivative, jnp.broadcast_to((weights + 1).sum(axis=0), (2, 3)))
    partial = uflx_jax_compiler.trace(expr, a, b, j)(locations, weights, 1)
    np.testing.assert_allclose(partial, expected[:, 1])


def test_real_contraction_of_integer_storage():
    i = index("i")
    a = RuntimeArray("a", (3,))
    values = np.full(3, 8, dtype=np.int8)
    assert uflx_jax_compiler.trace(index_sum(a[i] * a[i], (i,)), a)(values) == 192.0


def test_contracted_forms_receive_their_own_slots():
    i = index("i")
    V = space("scalar", None, scalar_type="real")
    u, v = function(V, "u"), function(V, "v")
    a = RuntimeArray("weights", (3,))
    left, right = form(a[i] * u, (u,)), form(v, (v,))
    total = index_sum(left * right, (i,))
    assert uflx_jax_compiler.trace(total, a)([1.0, 2.0, 3.0], 2.0, 5.0) == 60.0
