# Copyright (C) 2026 Jack S. Hale
#
# This file is part of uflx-core-experimental (https://fenics.github.io/uflx/language/)
#
# SPDX-License-Identifier:    MIT

# %% [markdown]
"""# Poisson on a triangle

Build a cell stiffness matrix and load vector for Poisson's equation with a
unit source. Define the quadratic basis functions and quadrature on the
reference triangle, then pass the physical vertices to the compiled kernels.
"""

# %%
import basix
import jax
import numpy as np

import uflx_jax_compiler
from uflx.calculus import gradient, integral
from uflx.forms import form, function
from uflx.geometry import create_mesh, euclidean_space
from uflx.passes import lower
from uflx_basix.cells import topology
from uflx_basix.quadrature import apply_quadrature
from uflx_basix.spaces import function_space

triangle = basix.CellType.triangle
T = topology(triangle)

p1 = basix.create_element(basix.ElementFamily.P, triangle, 1)
X = function_space(T, p1, shape=(T.dimension,))
x = function(X, "x")
mesh = create_mesh(x, euclidean_space(T.dimension))

p2 = basix.create_element(basix.ElementFamily.P, triangle, 2)
V = function_space(mesh, p2)
u, v = function(V, "u"), function(V, "v")

# %% [markdown]
# Write the stiffness and load integrals on the physical mesh.

# %%
a = integral(gradient(u, mesh) @ gradient(v, mesh), mesh)
L = integral(v, mesh)
# Lower to core normal form on the reference cell.
a_ref = lower(a)
L_ref = lower(L)

# %% [markdown]
# Degree-two quadrature is exact for both integrands on an affine triangle.

# %%
rule = basix.make_quadrature(triangle, degree=2)
# Discretise the continuous integral with quadrature
a_quadrature = apply_quadrature(a_ref, *rule)
L_quadrature = apply_quadrature(L_ref, *rule)

# Bind the physical test and trial fields to the local tensor axes.
a = form(a_quadrature, (v, u))
L = form(L_quadrature, (v,))

# The resulting kernels support JAX transforms such as jit, grad and vmap.
# Performance is one-order of magnitude slower than FFCx.
load = jax.jit(uflx_jax_compiler.trace(L, x))
stiffness = jax.jit(uflx_jax_compiler.trace(a, x))

# %% [markdown]
# Evaluate both kernels on a physical triangle. For a unit source, the six load
# entries sum to its area, 5.5.

# %%
vertices = np.array([(1.0, 1.0), (4.0, 2.0), (2.0, 5.0)])
b = load(vertices)  # cell load vector
A = stiffness(vertices)  # cell stiffness matrix

print("Cell load vector", b)
print("Cell stiffness matrix", A)
