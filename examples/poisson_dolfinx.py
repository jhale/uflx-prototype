# %% [markdown]
"""# Poisson with DOLFINx and EmitC

Use P1 solution and coefficient spaces on a unit square with P2 triangle geometry.
"""

# %%
import basix
import jax
import numpy as np
from dolfinx import fem, la, mesh
from dolfinx.la.superlu_dist import superlu_dist_matrix, superlu_dist_solver
from mpi4py import MPI

import uflx_dolfinx
from uflx.calculus import gradient, integral
from uflx.forms import form, function
from uflx.passes import lower
from uflx_basix.quadrature import apply_quadrature

jax.config.update("jax_enable_x64", True)

square = mesh.create_unit_square(MPI.COMM_WORLD, 8, 8)
p2 = basix.create_element(basix.ElementFamily.P, basix.CellType.triangle, 2)
square = fem.interpolate_geometry(square, fem.coordinate_element(p2))
V = fem.functionspace(square, ("Lagrange", 1))

u = function(V, "u")
v = function(V, "v")
domain = v.domain
kappa = uflx_dolfinx.Function(V, name="kappa")
kappa.interpolate(lambda X: 3 - 2 * X[0])
kappa.x.scatter_forward()

rule = basix.make_quadrature(square.basix_cell(), degree=1)
a = integral(kappa * (gradient(v, domain) @ gradient(u, domain)), domain)
a = form(apply_quadrature(lower(a), *rule), (v, u))
stiffness = uflx_dolfinx.form(a)

f = uflx_dolfinx.Constant(square, np.float64(2), name="f")
L = form(apply_quadrature(lower(integral(f * v, domain)), *rule), (v,))
load = uflx_dolfinx.form(L)

boundary = fem.locate_dofs_geometrical(V, lambda X: np.isclose(X[0], 0) | np.isclose(X[0], 1))
boundary_values = fem.Function(V)
boundary_values.interpolate(lambda X: X[0])
bc = fem.dirichletbc(boundary_values, boundary)

A = fem.assemble_matrix(stiffness, bcs=[bc])
A.scatter_reverse()
b = fem.assemble_vector(load)
fem.apply_lifting(b.array, [stiffness], bcs=[[bc]])
b.scatter_reverse(la.InsertMode.add)
bc.set(b.array)

solution = fem.Function(V)
solver = superlu_dist_solver(superlu_dist_matrix(A))
info = solver.solve(b, solution.x)
if info != 0:
    raise RuntimeError("SuperLU_DIST solve failed.")
solution.x.scatter_forward()

x_nodes = V.tabulate_dof_coordinates()[:, 0]
np.testing.assert_allclose(solution.x.array, x_nodes, atol=1e-12, rtol=0)
