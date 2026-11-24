# Copyright (C) 2026 Jack S. Hale
#
# This file is part of uflx-core-experimental (https://fenics.github.io/uflx/language/)
#
# SPDX-License-Identifier:    MIT

# %% [markdown]
"""# Durán–Liberman shear reduction

Cell-local Durán-Liberman shear reduction on an affine physical triangle.

We use the FEniCS-Shells convention grad(w) - theta. The lowest-order Basix
Nédélec element samples the tangential component at each edge midpoint. This
uses midpoint quadrature for the edge integrals of a quadratic rotation.

Use the interpolated rotation to compute the reduced Reissner–Mindlin shear energy.
"""

# %%
import basix
import jax
import numpy as np

from uflx.arithmetic import num
from uflx.calculus import gradient, integral
from uflx.forms import function
from uflx.geometry import create_mesh, euclidean_space, pull_back, push_forward
from uflx.passes import lower
from uflx_basix.cells import topology
from uflx_basix.interpolation import interpolate
from uflx_basix.quadrature import apply_quadrature
from uflx_basix.spaces import local_space
from uflx_jax_compiler import trace

triangle = basix.CellType.triangle
T = topology(triangle)
P1 = basix.create_element(basix.ElementFamily.P, triangle, 1)
P2 = basix.create_element(basix.ElementFamily.P, triangle, 2)
N1E = basix.create_element(basix.ElementFamily.N1E, triangle, 1)

x_ref = function(local_space(T, P1, shape=(2,)), "x")
mesh = create_mesh(x_ref, euclidean_space(2))
W = local_space(T, P1)
R = local_space(T, P2, shape=(2,))
shear_space = local_space(T, N1E)

w_ref = function(W, "w")  # Transverse displacement
theta_ref = function(R, "theta")  # Rotation
w = push_forward(w_ref, mesh, W.law)
theta = push_forward(theta_ref, mesh, R.law)

# %% [markdown]
# Each Nédélec degree of freedom samples the rotation along the mapped edge
# tangent at its midpoint.

# %%
reduced_rotation = interpolate(theta, shear_space)
reduced_shear = gradient(w, mesh) - reduced_rotation
shear_energy = integral((reduced_shear @ reduced_shear) / num(2), mesh)

energy_ref = lower(shear_energy)
# On an affine triangle the reduced shear is affine, so degree 2 integrates its energy exactly.
rule = basix.make_quadrature(triangle, degree=2)
energy_quadrature = apply_quadrature(energy_ref, *rule)
cell_energy = jax.jit(trace(energy_quadrature, x_ref, w_ref, theta_ref))

cell = mesh[0]
shear_ref = lower(pull_back(reduced_shear, cell))
shear_at = jax.jit(trace(shear_ref, x_ref, w_ref, theta_ref, cell.reference_cell))


# The cell map is (r, s) -> (2r + 2s, s). Supply physical nodal values
# of w = x and theta = (x² - y, x) on the skew triangle.
def physical_nodes(element):
    r, s = element.points.T
    return np.column_stack((2 * r + 2 * s, s))


vertices = physical_nodes(P1)
w_values = vertices[:, 0]
x, y = physical_nodes(P2).T
theta_values = np.column_stack((x**2 - y, x))
reference_point = np.array([0.25, 0.5])
point = vertices[0] + reference_point @ (vertices[1:] - vertices[0])
gamma = shear_at(vertices, w_values, theta_values, reference_point)

# %% [markdown]
# On each edge, the rotation has the same tangential component at the midpoint as `(1 - y, x)`.

# %%
# The physical reduced shear is therefore (y, -x).
energy = cell_energy(vertices, w_values, theta_values)
np.testing.assert_allclose(gamma, [point[1], -point[0]], atol=1e-6)
np.testing.assert_allclose(energy, 13 / 12, rtol=1e-6)
print("Reduced shear at", point, gamma)
print("Cell shear energy", energy)
