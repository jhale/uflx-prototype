# Copyright (C) 2026 Jack S. Hale
#
# This file is part of uflx-core-experimental (https://fenics.github.io/uflx/language/)
#
# SPDX-License-Identifier:    MIT

# %% [markdown]
"""# Precomputed geometry

Compute the inverse Jacobian and integration density at each quadrature point
on a curved P2 triangle. Reuse those arrays to assemble stiffness matrices for
two diffusivity fields on the same cell.
"""

# %%
import basix
import jax
import jax.numpy as jnp
import numpy as np

from uflx.algorithms import substitute_values
from uflx.calculus import gradient, integral
from uflx.forms import form, function
from uflx.geometry import create_mesh, density, euclidean_space, jacobian, push_forward
from uflx.passes import lower
from uflx.tensors import inverse
from uflx_basix.cells import topology
from uflx_basix.quadrature import apply_quadrature
from uflx_basix.spaces import local_space
from uflx_extensions.runtime_arrays import runtime_array
from uflx_jax_compiler import trace

triangle = basix.CellType.triangle
T = topology(triangle)
p2 = basix.create_element(basix.ElementFamily.P, triangle, 2)
s, t = p2.points.T
coordinates = np.column_stack((s, t + 0.4 * s * t))
ambient = euclidean_space(coordinates.shape[1])
x = function(local_space(T, p2, shape=(ambient.dimension,)), "x")
mesh = create_mesh(x, ambient)
cell = mesh[0]
K = cell.reference_cell

V = local_space(T, p2)
u_ref, v_ref, k_ref = (function(V, name) for name in ("u", "v", "k"))
u, v, k = (push_forward(f, mesh, V.law) for f in (u_ref, v_ref, k_ref))
a_ref = lower(integral(k * (gradient(u, mesh) @ gradient(v, mesh)), mesh))

# %% [markdown]
# ## Geometry kernel

# %%
invJ = inverse(jacobian(cell))
rho = density(cell)
invJ_at = trace(invJ, x, K)
rho_at = trace(rho, x, K)


def geometry_at_point(coordinates, point):
    return invJ_at(coordinates, point), rho_at(coordinates, point)


geometry = jax.jit(jax.vmap(geometry_at_point, in_axes=(None, 0)))

# %% [markdown]
# ## Assembly kernel
#
# Replace the two geometry expressions in the lowered integrand. Each point
# receives its own inverse Jacobian and density, with the same diffusivity
# DOFs used across the cell.

# %%
invJ_input = runtime_array("invJ", invJ.value_shape)
rho_input = runtime_array("rho", rho.value_shape)
integrand = substitute_values(a_ref.expression, {invJ: invJ_input, rho: rho_input})
slots = (v_ref, u_ref)
at_point = trace(form(integrand, slots), K, invJ_input, rho_input, k_ref)
at_points = jax.vmap(at_point, in_axes=(0, 0, 0, None))


@jax.jit
def assemble(points, weights, inverse_jacobians, densities, diffusivity):
    samples = at_points(points, inverse_jacobians, densities, diffusivity)
    return jnp.tensordot(weights, samples, axes=1)


# %%
points, weights = basix.make_quadrature(triangle, degree=4)
inverse_jacobians, densities = geometry(coordinates, points)

original = form(apply_quadrature(a_ref, points, weights), slots)
assemble_original = jax.jit(trace(original, x, k_ref))
for diffusivity in (1 + s, 1 + t):
    matrix = assemble(points, weights, inverse_jacobians, densities, diffusivity)
    expected = assemble_original(coordinates, diffusivity)
    np.testing.assert_allclose(matrix, expected, rtol=1e-5, atol=1e-6)
    print("Cell stiffness matrix", matrix)
