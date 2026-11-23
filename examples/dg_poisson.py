"""Symmetric interior penalty DG Poisson on a triangle and a quadrilateral.

The triangle carries a discontinuous P1 field on P2 geometry. The quadrilateral
carries a discontinuous Q1 field on Q2 geometry. Their shared edge is curved.
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

# Local finite elements

triangle = basix.CellType.triangle
quadrilateral = basix.CellType.quadrilateral
interval = basix.CellType.interval
family = basix.ElementFamily.P
p2 = basix.create_element(family, triangle, 2)
q2 = basix.create_element(family, quadrilateral, 2)
p1_dg = basix.create_element(family, triangle, 1, discontinuous=True)
q1_dg = basix.create_element(family, quadrilateral, 1, discontinuous=True)
plane = EuclideanSpace(2)

# Volume integral on each cell type

cell_integrals = {}
for cell_type, geometry_element, field_element in (
    (triangle, p2, p1_dg),
    (quadrilateral, q2, q1_dg),
):
    T = topology(cell_type)
    X = local_space(T, geometry_element, shape=(2,))
    x = function(X, "x")
    cell_mesh = Mesh(x, plane)
    V = local_space(T, field_element)
    cell_u_ref = function(V, "u")
    cell_v_ref = function(V, "v")
    cell_u = push_forward(cell_u_ref, cell_mesh, V.law)
    cell_v = push_forward(cell_v_ref, cell_mesh, V.law)
    grad_u = gradient(cell_u, cell_mesh)
    grad_v = gradient(cell_v, cell_mesh)
    volume = integral(grad_u @ grad_v, cell_mesh)
    cell_integrals[cell_type] = (volume, (cell_v_ref, cell_u_ref), x)

# Two-cell mesh and shared interval

T_pair = topology(triangle, quadrilateral)
X_pair = local_space(T_pair, (p2, q2), shape=(2,))
x_pair = function(X_pair, "x")
mesh = Mesh(x_pair, plane)
V_pair = local_space(T_pair, (p1_dg, q1_dg))
u_ref = function(V_pair, "u")
v_ref = function(V_pair, "v")
u = push_forward(u_ref, mesh, V_pair.law)
v = push_forward(v_ref, mesh, V_pair.law)

# Each cell has its own local facet numbering. Both sides use the two interval orientations.
triangle_facet = index_table("triangle_facet", shape=(), extent=3)
quad_facet = index_table("quad_facet", shape=(), extent=4)
local_facets = (triangle_facet[()], quad_facet[()])
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

# Lower and tabulate the cell integrals on their respective reference cells

cell_kernels = {}
for cell_type, (volume, arguments, x) in cell_integrals.items():
    volume_ref = lower(volume)
    cell_rule = basix.make_quadrature(cell_type, degree=8)
    volume_sum = apply_quadrature(volume_ref, *cell_rule)
    cell_form = form(volume_sum, arguments)
    evaluate_cell = trace(cell_form, x)
    cell_kernels[cell_type] = jax.jit(evaluate_cell)

# Lower the interior facet integrals on the reference interval

h_ref = lower(h)
flux_ref = lower(flux)
jumps_ref = lower(jumps)

# The arc length and the DG terms have independent quadrature rules.
length_rule = basix.make_quadrature(interval, degree=8)
facet_rule = basix.make_quadrature(interval, degree=8)
length = apply_quadrature(h_ref, *length_rule)
flux_sum = apply_quadrature(flux_ref, *facet_rule)
jump_sum = apply_quadrature(jumps_ref, *facet_rule)
penalty_weight = num(10) / length
facet_sum = flux_sum + penalty_weight * jump_sum
interior_facet_form = form(facet_sum, (v_ref, u_ref))
evaluate_facet = trace(interior_facet_form, x_pair, triangle_facet, quad_facet, permutations)


@jax.jit
def tabulate_interior_facet_tensor(coordinate_dofs, entity_local_index, quadrature_permutation):
    """Return the 7 x 7 tensor with triangle DOFs first and quadrilateral DOFs second."""
    plus_facet = entity_local_index[0]
    minus_facet = entity_local_index[1]
    return evaluate_facet(coordinate_dofs, plus_facet, minus_facet, quadrature_permutation)


# Gather the P2 and Q2 coordinate DOFs

# The triangle lies below the shared edge y = 0.32*x*(1-x) and the quadrilateral above it.

s, t = p2.points.T
triangle_dofs = np.column_stack((1 - s - t, -t + 0.32 * s * (1 - s - t)))

s, t = q2.points.T
quad_dofs = np.column_stack((s, t + 0.32 * s * (1 - s) * (1 - t)))

# Pack six triangle nodes followed by nine quadrilateral nodes.
coordinate_dofs = np.concatenate((triangle_dofs, quad_dofs))

# Triangle facet 2 runs right to left. Quadrilateral facet 0 runs left to right.
entity_local_index = np.array([2, 0], dtype=np.intc)
quadrature_permutation = np.array([1, 0], dtype=np.uint8)

# Tabulate the local tensors

triangle_tensor = cell_kernels[triangle](triangle_dofs)
quad_tensor = cell_kernels[quadrilateral](quad_dofs)
facet_tensor = tabulate_interior_facet_tensor(
    coordinate_dofs, entity_local_index, quadrature_permutation
)
print("Triangle cell tensor", triangle_tensor)
print("Quadrilateral cell tensor", quad_tensor)
print("Interior facet tensor", facet_tensor)
