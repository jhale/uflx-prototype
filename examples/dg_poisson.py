"""P1 symmetric interior penalty DG Poisson on two curved P2 triangles.

For this bilinear form the cell integral gives a 3 x 3 local tensor. The interior
facet integral couples the DOFs on both cells in a 6 x 6 local tensor.
"""

import basix
import jax
import numpy as np
from _interior_facet import interior_facet

from uflx.arithmetic import num
from uflx.calculus import gradient, integral
from uflx.forms import form, function
from uflx.geometry import pull_back, push_forward
from uflx.nodes import EuclideanSpace, Mesh
from uflx.passes import lower
from uflx_basix.cells import topology
from uflx_basix.quadrature import apply_quadrature
from uflx_basix.spaces import local_space
from uflx_extensions.index_tables import index_table
from uflx_jax_compiler import trace

# Cell stiffness on one triangle

triangle = basix.CellType.triangle
interval = basix.CellType.interval
p2 = basix.create_element(basix.ElementFamily.P, triangle, 2)
dg1 = basix.create_element(basix.ElementFamily.P, triangle, 1, discontinuous=True)

T = topology(triangle)
X = local_space(T, p2, shape=(2,))
x = function(X, "x")
plane = EuclideanSpace(2)
cell_mesh = Mesh(x, plane)

V = local_space(T, dg1)
cell_u_ref = function(V, "u")
cell_v_ref = function(V, "v")
cell_u = push_forward(cell_u_ref, cell_mesh, V.law)
cell_v = push_forward(cell_v_ref, cell_mesh, V.law)

cell_grad_u = gradient(cell_u, cell_mesh)
cell_grad_v = gradient(cell_v, cell_mesh)
volume = integral(cell_grad_u @ cell_grad_v, cell_mesh)

# Two-cell mesh and shared edge
T_pair = topology(triangle, triangle)
X_pair = local_space(T_pair, p2, shape=(2,))
x_pair = function(X_pair, "x")
mesh = Mesh(x_pair, plane)

V_pair = local_space(T_pair, dg1)
u_ref = function(V_pair, "u")
v_ref = function(V_pair, "v")
u = push_forward(u_ref, mesh, V_pair.law)
v = push_forward(v_ref, mesh, V_pair.law)

# On each side the assembler selects one of three triangle facets and one of two orientations.
local_facets = index_table("entity_local_index", shape=(2,), extent=3)
permutations = index_table("quadrature_permutation", shape=(2,), extent=2)
edge, plus, minus, n = interior_facet(mesh, local_facets, permutations)

# Interior penalty terms

jump_u = pull_back(u, plus) - pull_back(u, minus)
jump_v = pull_back(v, plus) - pull_back(v, minus)

grad_u = gradient(u, mesh)
grad_u_plus = pull_back(grad_u, plus)
grad_u_minus = pull_back(grad_u, minus)
avg_grad_u = (grad_u_plus + grad_u_minus) / num(2)

grad_v = gradient(v, mesh)
grad_v_plus = pull_back(grad_v, plus)
grad_v_minus = pull_back(grad_v, minus)
avg_grad_v = (grad_v_plus + grad_v_minus) / num(2)

# h is the full arc length, so the scalar 10/h can multiply the integrated jump product.
h = integral(num(1), edge)
normal_flux_u = avg_grad_u @ n
normal_flux_v = avg_grad_v @ n
flux = integral(-normal_flux_u * jump_v - normal_flux_v * jump_u, edge)
jumps = integral(jump_u * jump_v, edge)

# Lowering to the reference triangle and interval

volume_ref = lower(volume)
h_ref = lower(h)
flux_ref = lower(flux)
jumps_ref = lower(jumps)

# Quadrature and local bilinear forms

cell_rule = basix.make_quadrature(triangle, degree=8)
cell_sum = apply_quadrature(volume_ref, *cell_rule)
cell_form = form(cell_sum, (cell_v_ref, cell_u_ref))

# Integrate the edge length independently of the flux and jump terms.
length_rule = basix.make_quadrature(interval, degree=8)
length = apply_quadrature(h_ref, *length_rule)

facet_rule = basix.make_quadrature(interval, degree=8)
flux_sum = apply_quadrature(flux_ref, *facet_rule)
jump_sum = apply_quadrature(jumps_ref, *facet_rule)
penalty_weight = num(10) / length
facet_sum = flux_sum + penalty_weight * jump_sum
interior_facet_form = form(facet_sum, (v_ref, u_ref))

# Cell and interior facet integral kernels

# The cell integral needs only coordinate DOFs.
evaluate_cell_form = trace(cell_form, x)
tabulate_cell_tensor = jax.jit(evaluate_cell_form)

evaluate_interior_facet_form = trace(interior_facet_form, x_pair, local_facets, permutations)


@jax.jit
def tabulate_interior_facet_tensor(coordinate_dofs, entity_local_index, quadrature_permutation):
    """Return the local 6 x 6 tensor with the plus cell's DOFs followed by the minus cell's."""
    packed_dofs = coordinate_dofs.reshape(-1, 2)
    return evaluate_interior_facet_form(packed_dofs, entity_local_index, quadrature_permutation)


# Gather kernel inputs as a DOLFINx assembler would

# The common edge joins vertices 1 and 2. Moving its shared P2 node (node 4)
# away from the midpoint (0.5, 0.5) bends the edge towards vertex 3.
vertices = [(0, 0), (1, 0), (0, 1), (1, 1)]
edge_nodes = [(0.56, 0.56), (0, 0.5), (0.5, 0), (1, 0.5), (0.5, 1)]
geometry_points = np.array(vertices + edge_nodes)

# Basix P2 order is three vertices, then one node per edge opposite each vertex.
geometry_dofmap = np.array([(0, 1, 2, 4, 5, 6), (2, 1, 3, 7, 8, 4)])
# Gather coordinates in (cell, coordinate DOF, component) order.
# This 2D JAX example uses two components per DOF. UFCx packs three.
coordinate_dofs = geometry_points[geometry_dofmap]

# The shared edge is facet 0 on the first cell and facet 2 on the second.
# Orient both charts from the smaller global vertex ID to the larger one.
entity_local_index = np.array([0, 2], dtype=np.intc)
triangle_edges = basix.cell.topology(triangle)[1]
edge_vertices = np.asarray(triangle_edges)
local_endpoints = edge_vertices[entity_local_index]
global_endpoints = np.take_along_axis(geometry_dofmap, local_endpoints, axis=1)
reversed_edges = global_endpoints[:, 0] > global_endpoints[:, 1]
quadrature_permutation = reversed_edges.astype(np.uint8)

# Tabulate the local tensors

tabulate_cell_batch = jax.vmap(tabulate_cell_tensor)
cell_tensors = tabulate_cell_batch(coordinate_dofs)
interior_facet_tensor = tabulate_interior_facet_tensor(
    coordinate_dofs, entity_local_index, quadrature_permutation
)
print("Cell tensors", cell_tensors)
print("Interior facet tensor", interior_facet_tensor)
