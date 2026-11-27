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

Assemble the mixed cell stiffness matrix induced by the reduced Reissner–Mindlin
shear energy.
"""

# %%
import basix
import jax
import numpy as np

import uflx_jax_compiler
from uflx.calculus import gradient, integral
from uflx.forms import form, function
from uflx.geometry import create_mesh, euclidean_space, push_forward
from uflx.passes import lower
from uflx_basix.cells import topology
from uflx_basix.interpolation import interpolate
from uflx_basix.quadrature import apply_quadrature
from uflx_basix.spaces import local_space

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

w_ref, theta_ref = function(W, "w"), function(R, "theta")
v_ref, eta_ref = function(W, "v"), function(R, "eta")

w = push_forward(w_ref, mesh, W.law)
theta = push_forward(theta_ref, mesh, R.law)
v = push_forward(v_ref, mesh, W.law)
eta = push_forward(eta_ref, mesh, R.law)

# %% [markdown]
# Write $\gamma(w,\theta) = \nabla w - I_{N1E}\theta$. Since $\gamma$ is linear,
# the second variation of $E = \tfrac12 \int |\gamma|^2$ pairs the reduced
# shears of the test and trial fields.

# %%
def reduced_shear(w, theta):
    return gradient(w, mesh) - interpolate(theta, shear_space)

shear_form = integral(reduced_shear(v, eta) @ reduced_shear(w, theta), mesh)
# Both reduced shears are affine on this triangle, so degree 2 integrates their product exactly.
rule = basix.make_quadrature(triangle, degree=2)
a = form(apply_quadrature(lower(shear_form), *rule), ((v_ref, eta_ref), (w_ref, theta_ref)))
cell_matrix = jax.jit(uflx_jax_compiler.trace(a, x_ref))


# The cell map is (r, s) -> (2r + 2s, s). Supply physical nodal values
# of w = x and theta = (x² - y, x) on the skew triangle.
def physical_nodes(element):
    r, s = element.points.T
    return np.column_stack((2 * r + 2 * s, s))


vertices = physical_nodes(P1)
w_values = vertices[:, 0]
x, y = physical_nodes(P2).T
theta_values = np.column_stack((x**2 - y, x))
A = cell_matrix(vertices)
print("Cell shear stiffness matrix", A)

# %% [markdown]
# Each matrix axis starts with the three displacement DOFs, followed by the
# two rotation components at each of the six P2 nodes. For these nodal values,
# the interpolated rotation is $(1-y, x)$, so the reduced shear is $(y, -x)$.
# Its energy on the physical triangle is $\tfrac12 \int (x^2 + y^2) = 13/12$.

# %%
dofs = np.concatenate((w_values, theta_values.ravel()))
assert A.shape == (15, 15)
np.testing.assert_allclose(A, A.T, atol=1e-6)
np.testing.assert_allclose(dofs @ A @ dofs / 2, 13 / 12, rtol=1e-6)
