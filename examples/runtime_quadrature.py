# Copyright (C) 2026 Jack S. Hale
#
# This file is part of uflx-core-experimental (https://fenics.github.io/uflx/language/)
#
# SPDX-License-Identifier:    MIT

# %% [markdown]
"""# Runtime quadrature

Assemble a P2 triangle stiffness matrix with two quadrature rules. Each call
supplies its points, weights and the Basix basis tables at those points.
"""

# %%
import basix
import jax
import numpy as np

from uflx.calculus import gradient, integral
from uflx.forms import form, function
from uflx.geometry import create_mesh, euclidean_space, push_forward
from uflx.passes import lower
from uflx_basix.cells import topology
from uflx_basix.quadrature import apply_quadrature
from uflx_basix.spaces import local_space
from uflx_extensions.runtime_arrays import RuntimeArray, RuntimeExtent
from uflx_jax_compiler import basis_tables, trace

triangle = basix.CellType.triangle
T = topology(triangle)
p1 = basix.create_element(basix.ElementFamily.P, triangle, 1)
p2 = basix.create_element(basix.ElementFamily.P, triangle, 2)
x = function(local_space(T, p1, shape=(2,)), "x")
mesh = create_mesh(x, euclidean_space(2))
V = local_space(T, p2)
u_ref, v_ref = function(V, "u"), function(V, "v")
u = push_forward(u_ref, mesh, V.law)
v = push_forward(v_ref, mesh, V.law)
a_ref = lower(integral(gradient(u, mesh) @ gradient(v, mesh), mesh))

# %%
n = RuntimeExtent("n")
points_input = RuntimeArray("points", (n, 2))
weights_input = RuntimeArray("weights", (n,))
weighted_sum = apply_quadrature(a_ref, points_input, weights_input)
a = form(weighted_sum, (v_ref, u_ref))
tables = basis_tables(a)
stiffness = jax.jit(trace(a, x, points_input, weights_input, *tables))

# %% [markdown]
# The stiffness integrand is quadratic on an affine triangle, so both rules
# give the same matrix as the fixed degree-two rule.

# %%
vertices = np.array([(1.0, 1.0), (4.0, 2.0), (2.0, 5.0)])
fixed_sum = apply_quadrature(a_ref, *basix.make_quadrature(triangle, degree=2))
expected = trace(form(fixed_sum, (v_ref, u_ref)), x)(vertices)

for degree in (2, 4):
    points, weights = basix.make_quadrature(triangle, degree)
    values = [table.element.tabulate(table.order, points) for table in tables]
    matrix = stiffness(vertices, points, weights, *values)
    np.testing.assert_allclose(matrix, expected, rtol=1e-5, atol=1e-6)
    print(f"Stiffness matrix with {len(weights)} quadrature points", matrix)
