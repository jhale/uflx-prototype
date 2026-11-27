# Copyright (C) 2026 Jack S. Hale
#
# This file is part of uflx-core-experimental (https://fenics.github.io/uflx/language/)
#
# SPDX-License-Identifier:    MIT

# %% [markdown]
"""# Mixed Poisson

Couple a Raviart–Thomas flux with a discontinuous scalar on a curved triangle.
Differentiate the mixed energy to obtain the cell matrix and load vector.
"""

# %%
import basix
import jax
import numpy as np

import uflx_jax_compiler
from uflx.arithmetic import num
from uflx.calculus import directional_derivative, divergence, integral
from uflx.forms import form, function
from uflx.geometry import create_mesh, euclidean_space
from uflx.passes import lower
from uflx.tensors import broadcast_to
from uflx_basix.cells import topology
from uflx_basix.quadrature import apply_quadrature
from uflx_basix.spaces import function_space, local_space, real_scalar_space

triangle = basix.CellType.triangle
T = topology(triangle)
P2 = basix.create_element(basix.ElementFamily.P, triangle, 2)
x = function(local_space(T, P2, shape=(2,)), "x")
mesh = create_mesh(x, euclidean_space(2))

# %% [markdown]
# Pair a degree-two Raviart–Thomas flux space (Basix convention) with a
# discontinuous degree-one Lagrange space.

# %%
RT = basix.create_element(basix.ElementFamily.RT, triangle, 2)
P1_DG = basix.create_element(basix.ElementFamily.P, triangle, 1, discontinuous=True)

V = function_space(mesh, RT)
Q = function_space(mesh, P1_DG)
sigma, u = function(V, "sigma"), function(Q, "u")
state = (sigma, u)
test = (function(V, "tau"), function(Q, "v"))
trial_direction = (function(V, "delta_sigma"), function(Q, "delta_u"))

# For -div(kappa grad u) = 1 with homogeneous Dirichlet data, sigma = kappa grad u.
# The positive diffusivity kappa is supplied at runtime.
kappa = function(real_scalar_space(), "kappa")
half = num(1) / num(2)
flux_energy = (half / kappa) * (sigma @ sigma)
# The multiplier u enforces div(sigma) + 1 = 0.
energy = integral(flux_energy + u * (divergence(sigma, mesh) + num(1)), mesh)


def residual(at):
    return directional_derivative(energy, state, at, test)


# %% [markdown]
# The residual satisfies `R(w) = A w - b`. Differentiating it gives the cell
# matrix, and evaluating `-R(0)` gives the load vector.

# %%
zero_state = (broadcast_to(num(0), V.value_shape), num(0))
hessian = directional_derivative(residual(state), state, zero_state, trial_direction)

residual_at_zero = residual(zero_state)
hessian_ref = lower(hessian)
residual_ref = lower(residual_at_zero)

# Degree 8 approximates the rational flux mass term on the curved cell.
matrix_rule = basix.make_quadrature(triangle, degree=8)
# Degree 3 integrates the unit-source load exactly.
load_rule = basix.make_quadrature(triangle, degree=3)

hessian_quadrature = apply_quadrature(hessian_ref, *matrix_rule)
residual_quadrature = apply_quadrature(residual_ref, *load_rule)

a = form(hessian_quadrature, (test, trial_direction))
L = form(-residual_quadrature, (test,))

cell_matrix = jax.jit(uflx_jax_compiler.trace(a, x, kappa))
# The residual at zero retains kappa in terms that evaluate to zero.
cell_load = jax.jit(uflx_jax_compiler.trace(L, x, kappa))

# %% [markdown]
# Sample the coordinate map `(s, t) → (s, t + 0.4 s t)` at the six P2 nodes.

# %%
s, t = P2.points.T
coordinates = np.column_stack((s, t + 0.4 * s * t))
diffusivity = 2.0

# Each axis contains the eight flux DOFs followed by the three scalar DOFs.
A = cell_matrix(coordinates, diffusivity)
b = cell_load(coordinates, diffusivity)
print("Cell matrix", A)
print("Cell load vector", b)
