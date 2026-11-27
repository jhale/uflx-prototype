# Copyright (C) 2026 Jack S. Hale
#
# This file is part of uflx-core-experimental (https://fenics.github.io/uflx/language/)
#
# SPDX-License-Identifier:    MIT

# %% [markdown]
"""# Discontinuous Galerkin Poisson

Tabulate cell and interior facet tensors for the symmetric interior penalty DG Poisson form.

The topology has two triangular cells and one quadrilateral cell. In cell order,
their coordinate elements are P1, P2 and Q2. The triangles share a straight
facet. The second triangle and the quadrilateral share both endpoints and the
midpoint of a curved facet. The triangle's exterior facet and the quadrilateral's
interior node can be displaced independently. The field elements are
discontinuous P1 on the triangles and discontinuous Q1 on the quadrilateral.

The example builds all six interior facet kernels and checks the energy of
u(x, y) = x on the three cells and two interior facets of this patch.

Pull the fields back from both sides of each shared facet to form jumps and
average fluxes. To run the downloaded script, save
{download}`_interior_facet.py <../examples/_interior_facet.py>` beside it.
"""

# %%
from itertools import combinations_with_replacement

import basix
import jax
import numpy as np
from _interior_facet import interior_facet

import uflx_jax_compiler
from uflx.arithmetic import num
from uflx.calculus import gradient, integral
from uflx.forms import form, function
from uflx.geometry import create_mesh, euclidean_space, pull_back
from uflx.passes import lower
from uflx_basix.cells import topology
from uflx_basix.quadrature import apply_quadrature
from uflx_basix.spaces import function_space
from uflx_extensions.index_tables import index_table

# Topological cells of the patch

triangle = basix.CellType.triangle
quadrilateral = basix.CellType.quadrilateral
interval = basix.CellType.interval
patch_topology = topology(triangle, triangle, quadrilateral)

# Coordinate and field elements assigned to each cell

coordinate_degrees = (1, 2, 2)

family = basix.ElementFamily.P
plane = euclidean_space(2)
elements_by_cell = []
for cell, degree in zip(patch_topology.cell_types, coordinate_degrees, strict=True):
    coordinate_element = basix.create_element(family, cell.cell_type, degree)
    field_element = basix.create_element(family, cell.cell_type, 1, discontinuous=True)
    elements_by_cell.append((coordinate_element, field_element))

# Cells with the same coordinate and field elements share a kernel key.
cell_keys = tuple(dict.fromkeys(elements_by_cell))

# %% [markdown]
# ## Cell kernels

# %%
cell_kernels = {}
for cell_key in cell_keys:
    coordinate_element, field_element = cell_key
    cell_type = coordinate_element.cell_type
    
    T = topology(cell_type)
    X = function_space(T, coordinate_element, shape=(2,))
    x = function(X, "x")
    mesh = create_mesh(x, plane)
    
    V = function_space(mesh, field_element)
    u = function(V, "u")
    v = function(V, "v")

    grad_u = gradient(u, mesh)
    grad_v = gradient(v, mesh)
    volume = integral(grad_u @ grad_v, mesh)

    volume_ref = lower(volume)
    cell_rule = basix.make_quadrature(cell_type, degree=8)
    volume_sum = apply_quadrature(volume_ref, *cell_rule)
    cell_form = form(volume_sum, (v, u))
    evaluate_cell = uflx_jax_compiler.trace(cell_form, x)
    cell_kernels[cell_key] = jax.jit(evaluate_cell)

# %% [markdown]
# ## Interior facet kernels
#
# Use the order of `cell_keys` to choose the plus and minus sides.

# %%
local_facets = index_table("entity_local_index", shape=(2,))
permutations = index_table("quadrature_permutation", shape=(2,))

# All pairs share a reference interval. Arc length and DG terms have independent rules.
length_rule = basix.make_quadrature(interval, degree=8)
facet_rule = basix.make_quadrature(interval, degree=8)

facet_kernels = {}
for plus_key, minus_key in combinations_with_replacement(cell_keys, 2):
    plus_geometry, plus_field = plus_key
    minus_geometry, minus_field = minus_key
    plus_type = plus_geometry.cell_type
    minus_type = minus_geometry.cell_type
    
    T = topology(plus_type, minus_type)
    X = function_space(T, (plus_geometry, minus_geometry), shape=(2,))
    
    x = function(X, "x")
    mesh = create_mesh(x, plane)
    
    V = function_space(mesh, (plus_field, minus_field))
    
    u = function(V, "u")
    v = function(V, "v")

    facet, plus, minus, n = interior_facet(mesh, local_facets, permutations)

    # Symmetric interior penalty form on the shared physical facet
    jump_u = pull_back(u, plus) - pull_back(u, minus)
    jump_v = pull_back(v, plus) - pull_back(v, minus)

    grad_u = gradient(u, mesh)
    grad_v = gradient(v, mesh)
    avg_grad_u = (pull_back(grad_u, plus) + pull_back(grad_u, minus)) / num(2)
    avg_grad_v = (pull_back(grad_v, plus) + pull_back(grad_v, minus)) / num(2)
    normal_flux_u = avg_grad_u @ n
    normal_flux_v = avg_grad_v @ n

    # Divide the penalty parameter by the full arc length of the shared facet.
    penalty_parameter = num(10)
    h = integral(num(1), facet)
    h_ref = lower(h)
    length = apply_quadrature(h_ref, *length_rule)

    consistency = -normal_flux_u * jump_v - normal_flux_v * jump_u
    penalty = (penalty_parameter / length) * jump_u * jump_v
    facet_integral = integral(consistency + penalty, facet)

    facet_ref = lower(facet_integral)
    facet_sum = apply_quadrature(facet_ref, *facet_rule)
    interior_facet_form = form(facet_sum, (v, u))
    # The selected facet and permutation determine the parent reference points at runtime.
    evaluate_facet = uflx_jax_compiler.trace(
        interior_facet_form, x, local_facets, permutations
    )
    facet_kernels[plus_key, minus_key] = jax.jit(evaluate_facet)

# Typical form compiler would hand off cell_kernels and facet_kernels to
# assembler at this point

# %% [markdown]
# ## Mesh geometry

# %%
# Set interface_offset to zero for a straight P2/Q2 facet. The other offsets
# are independent.
boundary_offset = 0.15
interface_offset = 0.08
quad_interior_offset = 0.03
nodes = np.array(
    [
        (0.0, 0.0),
        (1.0, 0.0),
        (0.0, 1.0),
        (-1.0, 0.0),
        (0.0, -1.0),
        (1.0, -1.0),
        (0.5, 0.5 + boundary_offset),  # Node 6 on the P2 exterior facet
        (0.0, 0.5),  # Node 7 lies halfway along the straight P1/P2 facet
        (0.5, interface_offset),  # Node 8 belongs to both sides of the curved P2/Q2 facet
        (0.5, -1.0),
        (0.0, -0.5),
        (1.0, -0.5),
        (0.5, -0.5 + quad_interior_offset),  # Node 12 affects only the quadrilateral interior
    ]
)

# Coordinate DOFs in Basix order. The P2 and Q2 maps give their shared facet opposite orientations.
geometry_dofmaps = (
    [2, 3, 0],
    [0, 1, 2, 6, 7, 8],
    [1, 0, 5, 4, 8, 11, 10, 9, 12],
)
cell_data = tuple(zip(elements_by_cell, geometry_dofmaps, strict=True))

# Each row gives the incident cells (+, -), their local facet numbers, and their permutations.
# Orient the straight facet from vertex 0 to 2 and the curved facet from vertex 0 to 1.
interior_facets = (
    ((0, 1), (1, 1), (1, 0)),
    ((1, 2), (2, 0), (0, 1)),
)

# %% [markdown]
# Tabulate the tensors and evaluate the patch energy for `u(x, y) = x`.

# %%
patch_energy = 0.0
x_dofs = []
for cell_index, (cell_key, dofs) in enumerate(cell_data):
    coordinates = nodes[dofs]
    cell_tensor = cell_kernels[cell_key](coordinates)
    print("Cell tensor", cell_index, cell_tensor)

    # The P1 and Q1 field DOFs are vertex values. Vertices come first in each geometry DOF map.
    _, field_element = cell_key
    cell_x = coordinates[: field_element.dim, 0]
    x_dofs.append(cell_x)
    patch_energy += cell_x @ cell_tensor @ cell_x

for (plus, minus), facet_numbers, permutation_codes in interior_facets:
    plus_key, plus_dofs = cell_data[plus]
    minus_key, minus_dofs = cell_data[minus]
    kernel = facet_kernels[plus_key, minus_key]
    coordinate_dofs = np.concatenate((nodes[plus_dofs], nodes[minus_dofs]))
    entity_local_index = np.asarray(facet_numbers, dtype=np.intc)
    quadrature_permutation = np.asarray(permutation_codes, dtype=np.uint8)
    facet_tensor = kernel(coordinate_dofs, entity_local_index, quadrature_permutation)
    print("Interior facet tensor", (plus, minus), facet_tensor)

    # The continuous field has zero jump on both interfaces.
    facet_x = np.concatenate((x_dofs[plus], x_dofs[minus]))
    facet_energy = facet_x @ facet_tensor @ facet_x
    np.testing.assert_allclose(facet_energy, 0.0, atol=1e-6)
    patch_energy += facet_energy

    # A unit jump costs 10 * h_facet / h_length. Both rules give the same facet length here.
    unit_jump = np.concatenate((np.ones_like(x_dofs[plus]), np.zeros_like(x_dofs[minus])))
    penalty_energy = unit_jump @ facet_tensor @ unit_jump
    np.testing.assert_allclose(penalty_energy, 10.0, rtol=1e-6)

# The squared gradient is one. Bending the interior facet transfers area between its neighbours.
# The exterior facet adds the area under 4 * boundary_offset * s * (1 - s) for s between 0 and 1.
expected_area = 2.0 + 2.0 * boundary_offset / 3.0
np.testing.assert_allclose(patch_energy, expected_area, rtol=1e-6)
print("Patch energy for u(x, y) = x", patch_energy)
