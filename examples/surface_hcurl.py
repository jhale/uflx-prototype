# Copyright (C) 2026 Jack S. Hale
#
# This file is part of uflx-core-experimental (https://fenics.github.io/uflx/language/)
#
# SPDX-License-Identifier:    MIT

# %% [markdown]
"""# A surface H(curl) mass matrix

Assemble the mass matrix of a lowest-order Nédélec space on two triangles
meeting along a fold in three dimensions. The shared degree of freedom is
the circulation along their common edge. Both cells must interpret its sign
with respect to the same edge direction.
"""

# %%
import basix
import jax
import jax.numpy as jnp
import numpy as np

from uflx.calculus import integral
from uflx.forms import form, function
from uflx.geometry import create_mesh, euclidean_space
from uflx.passes import lower
from uflx_basix.cells import create_topology
from uflx_basix.quadrature import apply_quadrature
from uflx_basix.spaces import function_space
from uflx_jax_compiler import trace

triangle = basix.CellType.triangle
topology = create_topology(triangle)
p1 = basix.create_element(basix.ElementFamily.P, triangle, 1)
x = function(function_space(topology, p1, shape=(3,)), "x")
mesh = create_mesh(x, euclidean_space(3))
nedelec = basix.create_element(basix.ElementFamily.N1E, triangle, 1)
V = function_space(mesh, nedelec)
u, v = function(V, "u"), function(V, "v")

# %% [markdown]
# Write the form on the physical mesh. Lowering maps each two-component
# reference basis function to a three-component tangential vector by
# the transpose of the Jacobian pseudoinverse.

# %%
mass = integral(v @ u, mesh)
rule = basix.make_quadrature(triangle, degree=2)
cell_matrix = trace(form(apply_quadrature(lower(mass), *rule), (v, u)), x)

# %% [markdown]
# Orient each global edge from the smaller vertex index to the larger.
# Basix gives the directed reference edges in local degree-of-freedom order
# for this element. On the shared edge the first triangle uses 1 → 2 and
# the second uses 2 → 1. Compute the local-to-global edge numbers and signs
# once from the face connectivity. They can be reused when the vertices move.

# %%
vertices = np.array([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (1.0, 1.0, 1.0)])
faces = np.array([(0, 1, 2), (2, 1, 3)])
directed_edges = faces[:, basix.topology(triangle)[1]]
signs = np.where(directed_edges[..., 0] < directed_edges[..., 1], 1, -1)
edges, indices = np.unique(
    np.sort(directed_edges, axis=-1).reshape(-1, 2), axis=0, return_inverse=True
)
dofs = indices.reshape(signs.shape)

# %% [markdown]
# With connectivity fixed, the compiled assembler takes only vertex coordinates.
# Apply the edge signs on both matrix axes and sum the shared entries.


# %%
@jax.jit
def assemble(vertices):
    local = jax.vmap(cell_matrix)(vertices[faces])
    oriented = signs[:, :, None] * local * signs[:, None, :]
    matrix = jnp.zeros((len(edges), len(edges)), dtype=local.dtype)
    return matrix.at[dofs[:, :, None], dofs[:, None, :]].add(oriented)


A = assemble(vertices)

# %% [markdown]
# For the potential φ = x + 2y, the circulation of its surface gradient
# along each oriented edge is the difference of endpoint values. This field
# belongs to the discrete space on both affine faces. Its squared norm is 5
# on the horizontal face and 2 on the face with normal (-1, -1, 1)/√3.
# Their areas are 1/2 and √3/2, respectively.

# %%
potential = vertices[:, 0] + 2 * vertices[:, 1]
circulations = potential[edges[:, 1]] - potential[edges[:, 0]]
energy = circulations @ A @ circulations
np.testing.assert_allclose(energy, 2.5 + np.sqrt(3), rtol=1e-6)
print("Surface gradient energy", energy)
