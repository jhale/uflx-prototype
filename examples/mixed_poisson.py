"""Mixed Poisson cell matrix and load vector on a curved quadratic triangle."""

import basix
import jax
import numpy as np

from uflx.arithmetic import num
from uflx.calculus import directional_derivative, divergence, integral
from uflx.forms import form, function
from uflx.geometry import create_mesh, euclidean_space, push_forward
from uflx.passes import lower
from uflx.tensors import broadcast_to
from uflx_basix.cells import topology
from uflx_basix.quadrature import apply_quadrature
from uflx_basix.spaces import local_space, real_scalar_space
from uflx_jax_compiler import trace

triangle = basix.CellType.triangle
T = topology(triangle)
P2 = basix.create_element(basix.ElementFamily.P, triangle, 2)
x = function(local_space(T, P2, shape=(2,)), "x")
mesh = create_mesh(x, euclidean_space(2))

# Basix degree 2 Raviart-Thomas and discontinuous degree 1 Lagrange.
RT = basix.create_element(basix.ElementFamily.RT, triangle, 2)
P1_DG = basix.create_element(basix.ElementFamily.P, triangle, 1, discontinuous=True)

V = local_space(T, RT)
Q = local_space(T, P1_DG)
sigma_ref, u_ref = function(V, "sigma"), function(Q, "u")
state = (sigma_ref, u_ref)
test = (function(V, "tau"), function(Q, "v"))
trial_direction = (function(V, "delta_sigma"), function(Q, "delta_u"))

# The RT flux uses the contravariant Piola map.
sigma = push_forward(sigma_ref, mesh, V.law)
# For the cell map F and reference point X, the DG scalar satisfies u(F(X)) = u_ref(X).
u = push_forward(u_ref, mesh, Q.law)

# For -div(kappa grad u) = 1 with homogeneous Dirichlet data, sigma = kappa grad u.
# The positive diffusivity kappa is supplied at runtime.
kappa = function(real_scalar_space(), "kappa")
half = num(1) / num(2)
flux_energy = (half / kappa) * (sigma @ sigma)
# The multiplier u enforces div(sigma) + 1 = 0.
energy = integral(flux_energy + u * (divergence(sigma, mesh) + num(1)), mesh)


def residual(at):
    return directional_derivative(energy, state, at, test)


# R(w) = A w - b, so differentiating R gives A and evaluating -R(0) gives b.
zero_state = (broadcast_to(num(0), V.value_shape), num(0))
hessian = directional_derivative(residual(state), state, zero_state, trial_direction)

residual_at_zero = residual(zero_state)
hessian_ref = lower(hessian)
residual_ref = lower(residual_at_zero)

# Degree 8 approximates the rational flux mass term on the curved cell.
matrix_rule = basix.make_quadrature(triangle, degree=8)
# Degree 3 integrates the unit-source load exactly.
load_rule = basix.make_quadrature(triangle, degree=3)

hessian_quadrature = apply_quadrature(hessian_ref, *matrix_rule)
residual_quadrature = apply_quadrature(residual_ref, *load_rule)

a = form(hessian_quadrature, (test, trial_direction))
L = form(-residual_quadrature, (test,))

cell_matrix = jax.jit(trace(a, x, kappa))
# The residual at zero retains kappa in terms that evaluate to zero.
cell_load = jax.jit(trace(L, x, kappa))

# Sample the coordinate map (s, t) -> (s, t + 0.4*s*t) at the six P2 nodes.
s, t = P2.points.T
coordinates = np.column_stack((s, t + 0.4 * s * t))
diffusivity = 2.0

# Each axis contains the eight flux DOFs followed by the three scalar DOFs.
A = cell_matrix(coordinates, diffusivity)
b = cell_load(coordinates, diffusivity)
print("Cell matrix", A)
print("Cell load vector", b)
