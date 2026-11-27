# Copyright (C) 2026 Jack S. Hale
#
# This file is part of uflx-core-experimental (https://fenics.github.io/uflx/language/)
#
# SPDX-License-Identifier:    MIT

# %% [markdown]
"""# Runtime quadrature

Assemble a P2 triangle stiffness matrix with two quadrature rules. Each call
supplies the reference points, weights and P2 basis gradients at those points.
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
from uflx.tensors import index
from uflx_basix.cells import topology
from uflx_basix.quadrature import apply_quadrature, sample_form, weighted_sum
from uflx_basix.spaces import function_space, local_space
from uflx_extensions.runtime_arrays import runtime_array, runtime_extent

triangle = basix.CellType.triangle
T = topology(triangle)
vertices = np.array([(1.0, 1.0), (4.0, 2.0), (2.0, 5.0)])
ambient = euclidean_space(vertices.shape[1])
p1 = basix.create_element(basix.ElementFamily.P, triangle, 1)
p2 = basix.create_element(basix.ElementFamily.P, triangle, 2)
x = function(local_space(T, p1, shape=(ambient.dimension,)), "x")
mesh = create_mesh(x, ambient)
V = function_space(mesh, p2)
u, v = function(V, "u"), function(V, "v")
a_ref = lower(integral(gradient(u, mesh) @ gradient(v, mesh), mesh))
cell = a_ref.integration_domain

# %%
n = runtime_extent("n")
points_input = runtime_array("points", (n, cell.dimension))
weights_input = runtime_array("weights", (n,))
q = index("q")

samples, tables = sample_form(form(a_ref, (v, u)), points_input, q)
p2_basis = tables[p2, 1]
cell_matrix = weighted_sum(samples, weights_input, q)
stiffness = jax.jit(uflx_jax_compiler.trace(cell_matrix, x, points_input, weights_input, p2_basis))

# %% [markdown]
# The stiffness integrand is quadratic on an affine triangle, so both rules
# give the same matrix as the fixed degree-two rule.

# %%
fixed_sum = apply_quadrature(a_ref, *basix.make_quadrature(triangle, degree=2))
expected = uflx_jax_compiler.trace(form(fixed_sum, (v, u)), x)(vertices)

for degree in (2, 4):
    points, weights = basix.make_quadrature(triangle, degree)
    matrix = stiffness(vertices, points, weights, p2.tabulate(1, points))
    np.testing.assert_allclose(matrix, expected, rtol=1e-5, atol=1e-6)
    print(f"Stiffness matrix with {len(weights)} quadrature points", matrix)
