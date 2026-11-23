"""P1 symmetric interior penalty DG Poisson on two curved P2 triangles."""

import basix
import jax
import numpy as np
from _interior_facet import facet_geometry_values, interior_facet

from uflx.arithmetic import num
from uflx.calculus import gradient, integral
from uflx.forms import form, function
from uflx.geometry import conormal, pull_back, push_forward
from uflx.nodes import EuclideanSpace, Mesh
from uflx.passes import lower
from uflx_basix.cells import topology
from uflx_basix.quadrature import apply_quadrature
from uflx_basix.spaces import local_space
from uflx_jax_compiler import trace

triangle, interval = basix.CellType.triangle, basix.CellType.interval
p2 = basix.create_element(basix.ElementFamily.P, triangle, 2)
dg1 = basix.create_element(basix.ElementFamily.P, triangle, 1, discontinuous=True)

# The volume integral lives on one triangle.
T = topology(triangle)
x = function(local_space(T, p2, shape=(2,)), "x")
cell_mesh = Mesh(x, EuclideanSpace(2))
V = local_space(T, dg1)
cell_u_ref, cell_v_ref = function(V, "u"), function(V, "v")
cell_u = push_forward(cell_u_ref, cell_mesh, V.law)
cell_v = push_forward(cell_v_ref, cell_mesh, V.law)
volume = integral(gradient(cell_u, cell_mesh) @ gradient(cell_v, cell_mesh), cell_mesh)

# The interior-facet integral lives on the common edge of two triangles.
T_pair = topology(triangle, triangle)
x_pair = function(local_space(T_pair, p2, shape=(2,)), "x")
mesh = Mesh(x_pair, EuclideanSpace(2))
V_pair = local_space(T_pair, dg1)
u_ref, v_ref = function(V_pair, "u"), function(V_pair, "v")
u = push_forward(u_ref, mesh, V_pair.law)
v = push_forward(v_ref, mesh, V_pair.law)
(edge, plus, minus), (plus_chart, minus_chart, ref_normal) = interior_facet(mesh[0], mesh[1])
n = conormal(plus, ref_normal)

jump_u = pull_back(u, plus) - pull_back(u, minus)
jump_v = pull_back(v, plus) - pull_back(v, minus)
grad_u, grad_v = gradient(u, mesh), gradient(v, mesh)
avg_grad_u = (pull_back(grad_u, plus) + pull_back(grad_u, minus)) / num(2)
avg_grad_v = (pull_back(grad_v, plus) + pull_back(grad_v, minus)) / num(2)

# h is the full arc length, so the scalar 10/h can multiply the integrated jump product.
h = integral(num(1), edge)
flux = integral(-(avg_grad_u @ n) * jump_v - (avg_grad_v @ n) * jump_u, edge)
jumps = integral(jump_u * jump_v, edge)

# Curvature makes the metric and normal vary. These rules approximate the integrals.
volume_ref = lower(volume)
h_ref, flux_ref, jumps_ref = lower(h), lower(flux), lower(jumps)
cell_rule = basix.make_quadrature(triangle, degree=8)
facet_rule = basix.make_quadrature(interval, degree=8)

cell_sum = apply_quadrature(volume_ref, *cell_rule)
length = apply_quadrature(h_ref, *facet_rule)
flux_sum = apply_quadrature(flux_ref, *facet_rule)
jump_sum = apply_quadrature(jumps_ref, *facet_rule)
cell_form = form(cell_sum, (cell_v_ref, cell_u_ref))
facet_form = form(flux_sum + (num(10) / length) * jump_sum, (v_ref, u_ref))

# Trace both forms. The JIT-compiled facet kernel selects the chart on each side at runtime.
cell_kernel = jax.jit(trace(cell_form, x))
facet_action = trace(facet_form, x_pair, plus_chart, minus_chart, ref_normal)


@jax.jit
def interior_facet_kernel(coordinates, local_facets, permutations):
    """Return the 6 x 6 matrix for two quadratic triangles sharing a curved edge."""
    plus_points, minus_points, normal = facet_geometry_values(local_facets, permutations)
    return facet_action(coordinates.reshape(-1, 2), plus_points, minus_points, normal)


# The common edge joins vertices 1 and 2. Moving its shared P2 node (node 4)
# away from the midpoint (0.5, 0.5) bends the edge towards vertex 3.
vertices = [(0, 0), (1, 0), (0, 1), (1, 1)]
edge_nodes = [(0.56, 0.56), (0, 0.5), (0.5, 0), (1, 0.5), (0.5, 1)]
nodes = np.array(vertices + edge_nodes)
# Basix P2 order is three vertices, then one node per edge opposite each vertex.
cell_nodes = np.array([(0, 1, 2, 4, 5, 6), (2, 1, 3, 7, 8, 4)])
coordinates = nodes[cell_nodes]
local_facets = np.array([0, 2])

# Orient the common edge from the smaller global vertex ID to the larger one.
edge_vertices = np.asarray(basix.cell.topology(triangle)[1])
start, end = np.take_along_axis(cell_nodes, edge_vertices[local_facets], axis=1).T
permutations = (start > end).astype(np.uint8)

cell_matrices = jax.vmap(cell_kernel)(coordinates)
facet_matrix = interior_facet_kernel(coordinates, local_facets, permutations)
print("Cell matrices", cell_matrices)
print("Interior facet matrix", facet_matrix)
