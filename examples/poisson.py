"""P2 Poisson cell tensors on the unit triangle."""

import basix
import jax
import numpy as np

from uflx.calculus import gradient, integral
from uflx.forms import form, function
from uflx.geometry import push_forward
from uflx.nodes import EuclideanSpace, Mesh
from uflx.passes import lower
from uflx_basix.cells import topology
from uflx_basix.quadrature import apply_quadrature
from uflx_basix.spaces import local_space
from uflx_jax_compiler import trace

triangle = basix.CellType.triangle
T = topology(triangle)

p1 = basix.create_element(basix.ElementFamily.P, triangle, 1)
X = local_space(T, p1, shape=(2,))
x = function(X, "x")
mesh = Mesh(x, EuclideanSpace(2))

p2 = basix.create_element(basix.ElementFamily.P, triangle, 2)
V = local_space(T, p2)
u_ref, v_ref = function(V, "u"), function(V, "v")
u = push_forward(u_ref, mesh, V.law)
v = push_forward(v_ref, mesh, V.law)

# We write integrals on the physical mesh using the pushed forward functions
a = integral(gradient(u, mesh) @ gradient(v, mesh), mesh)
L = integral(v, mesh)
# and lower then brings them to core normal form. 
a_ref = lower(a)
L_ref = lower(L)

# On an affine triangle both integrands are quadratic, so degree 2 is exact.
rule = basix.make_quadrature(triangle, degree=2)
# Rewrite the continuous integral with quadrature
a_quadrature = apply_quadrature(a_ref, *rule)
L_quadrature = apply_quadrature(L_ref, *rule)

# Note that the form arguments are the basis functions on the reference cell.
a = form(a_quadrature, (v_ref, u_ref))
L = form(L_quadrature, (v_ref,))

# Trace to JAX and JIT compile. All standard JAX transforms are supported, e.g.
# differentiation, mapping etc.
load = jax.jit(trace(L, x))
stiffness = jax.jit(trace(a, x))

vertices = np.array([(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)])
b = load(vertices)  # cell load vector
A = stiffness(vertices)  # cell stiffness matrix

print("Cell load vector", b)
print("Cell stiffness matrix", A)
