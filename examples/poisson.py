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

# Note that inside lower we are writing intergrals on the physical mesh...
a_ref = lower(integral(gradient(u, mesh) @ gradient(v, mesh), mesh))
L_ref = lower(integral(v, mesh))

# On an affine triangle both integrands are quadratic, so degree 2 is exact.
rule = basix.make_quadrature(triangle, degree=2)

# ...but after pullback (lower), the arguments of the form live on the
# reference element, hence v_ref and u_ref are used.
a = form(apply_quadrature(a_ref, *rule), (v_ref, u_ref))
L = form(apply_quadrature(L_ref, *rule), (v_ref,))

load = jax.jit(trace(L, x))
stiffness = jax.jit(trace(a, x))

vertices = np.array([(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)])
b = load(vertices)  # cell load vector
A = stiffness(vertices)  # cell stiffness matrix

print("Cell load vector", b)
print("Cell stiffness matrix", A)
