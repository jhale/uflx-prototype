# Copyright (C) 2026 Jack S. Hale
#
# This file is part of uflx-core-experimental (https://fenics.github.io/uflx/language/)
#
# SPDX-License-Identifier:    MIT

# %% [markdown]
"""# Precomputed geometry

Precompute the inverse Jacobian and integration density at quadrature points
on a curved triangle, then use them to assemble its P2 stiffness matrix.
"""

# %%
import basix
import jax
import jax.numpy as jnp

from uflx.algorithms import substitute_values
from uflx.calculus import gradient, integral
from uflx.forms import form, function
from uflx.geometry import create_mesh, density, euclidean_space, jacobian, push_forward
from uflx.passes import lower
from uflx.tensors import inverse
from uflx_basix.cells import topology
from uflx_basix.spaces import local_space
from uflx_extensions.runtime_arrays import runtime_array
from uflx_jax_compiler import trace

triangle = basix.CellType.triangle
cell_topology = topology(triangle)
p2 = basix.create_element(basix.ElementFamily.P, triangle, 2)
s, t = p2.points.T
coordinates = jnp.column_stack((s, t + 0.4 * s * t))
ambient = euclidean_space(coordinates.shape[1])
x = function(local_space(cell_topology, p2, shape=(ambient.dimension,)), "x")
mesh = create_mesh(x, ambient)
cell_map = mesh[0]
reference_cell = cell_map.reference_cell

V = local_space(cell_topology, p2)
u_ref, v_ref = function(V, "u"), function(V, "v")
u, v = (push_forward(f, mesh, V.law) for f in (u_ref, v_ref))
stiffness_integral = lower(integral(gradient(u, mesh) @ gradient(v, mesh), mesh))

# %%
inverse_jacobian = inverse(jacobian(cell_map))
integration_density = density(cell_map)
evaluate_inverse_jacobian = trace(inverse_jacobian, x, reference_cell)
evaluate_density = trace(integration_density, x, reference_cell)


def geometry_at_point(coordinates, point):
    return evaluate_inverse_jacobian(coordinates, point), evaluate_density(coordinates, point)


precompute_geometry = jax.jit(jax.vmap(geometry_at_point, in_axes=(None, 0)))

# %%
inverse_jacobian_input = runtime_array("inverse_jacobian", inverse_jacobian.value_shape)
density_input = runtime_array("integration_density", integration_density.value_shape)
integrand = substitute_values(
    stiffness_integral.expression,
    {inverse_jacobian: inverse_jacobian_input, integration_density: density_input},
)
evaluate_integrand = trace(
    form(integrand, (v_ref, u_ref)), reference_cell, inverse_jacobian_input, density_input
)


@jax.jit
def assemble_stiffness(points, weights, inverse_jacobians, integration_densities):
    samples = jax.vmap(evaluate_integrand)(points, inverse_jacobians, integration_densities)
    return jnp.tensordot(weights, samples, axes=1)


# %%
points, weights = basix.make_quadrature(triangle, degree=4)
inverse_jacobians, integration_densities = precompute_geometry(coordinates, points)
matrix = assemble_stiffness(points, weights, inverse_jacobians, integration_densities)
print("Cell stiffness matrix", matrix)
