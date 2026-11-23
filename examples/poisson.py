import basix
import numpy as np

from uflx.calculus import gradient, integral
from uflx.forms import form, function
from uflx.geometry import push_forward
from uflx.nodes import EuclideanSpace, Mesh
from uflx_basix.cells import topology
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

a = form(integral(gradient(u, mesh) @ gradient(v, mesh), mesh), (v_ref, u_ref))
L = form(integral(v, mesh), (v_ref,))

load = trace(L, x)

# Supply the coordinate values and the P2 test-function values.
vertices = np.array([(0.0, 0.0), (2.0, 0.0), (0.0, 1.0)])
area = load(vertices, np.ones(p2.dim))  # L(1) is the area, 1.
print(f"L(1) = {area:g}")
