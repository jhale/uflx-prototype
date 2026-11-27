# Copyright (C) 2026 Jack S. Hale
#
# This file is part of uflx-core-experimental (https://fenics.github.io/uflx/language/)
#
# SPDX-License-Identifier:    MIT

# %% [markdown]
"""# Precomputed geometry

Extract the geometry terms from the physical Poisson integral on a curved
triangle. Evaluate them at quadrature points and assemble the stiffness
matrix in one JIT-compiled kernel.
"""

# %%
import basix
import jax
import numpy as np

import uflx_jax_compiler
from uflx.algorithms import substitute_values
from uflx.calculus import gradient, integral
from uflx.forms import form, function
from uflx.geometry import create_mesh, euclidean_space
from uflx.passes import lower
from uflx.tensors import index
from uflx_basix.cells import topology
from uflx_basix.evaluation import evaluate_at_points
from uflx_basix.quadrature import weighted_sum
from uflx_basix.spaces import function_space, local_space
from uflx_extensions.runtime_arrays import runtime_array

triangle = basix.CellType.triangle
cell_topology = topology(triangle)
p2 = basix.create_element(basix.ElementFamily.P, triangle, 2)

ambient = euclidean_space(cell_topology.dimension)
x = function(local_space(cell_topology, p2, shape=(ambient.dimension,)), "x")
mesh = create_mesh(x, ambient)

V = function_space(mesh, p2)
u, v = function(V, "u"), function(V, "v")

stiffness = lower(integral(gradient(u, mesh) @ gradient(v, mesh), mesh))
stiffness, geometry = uflx_jax_compiler.extract_geometry(stiffness, mesh[0])

print(geometry)

# %% [markdown]
# ## Geometry at quadrature points

# %%
points, weights = basix.make_quadrature(triangle, degree=4)
q = index("q")

evaluators = [
    uflx_jax_compiler.trace(evaluate_at_points(term, points, q), x) for term in geometry.values()
]

# %% [markdown]
# ## Stiffness assembly

# %%
tables = {
    value: runtime_array(value.label, (len(points), *value.value_shape)) for value in geometry
}
integrand = substitute_values(
    stiffness.expression, {value: table[q] for value, table in tables.items()}
)

samples = evaluate_at_points(integrand, points, q)
a = form(weighted_sum(samples, weights, q), (v, u))

assemble_stiffness = uflx_jax_compiler.trace(a, *tables.values())


@jax.jit
def cell_stiffness(coordinates):
    geometry_values = tuple(evaluate(coordinates) for evaluate in evaluators)
    return assemble_stiffness(*geometry_values)


s, t = p2.points.T
coordinates = np.column_stack((s, t + 0.4 * s * t))
matrix = cell_stiffness(coordinates)
print("Cell stiffness matrix", matrix)
