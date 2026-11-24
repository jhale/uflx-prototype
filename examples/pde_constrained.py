# Copyright (C) 2026 Jack S. Hale
#
# This file is part of uflx-core-experimental (https://fenics.github.io/uflx/language/)
#
# SPDX-License-Identifier:    MIT

# %% [markdown]
"""# PDE-constrained optimisation

Forms for control of a reaction coefficient on the unit interval.

A unit source drives a reaction-diffusion equation with zero flux at both ends.
We optimise the positive reaction coefficient m to bring the state u close to
the target value one.

Directional derivatives can be nested symbolically. To inspect the expanded
form, call apply_derivatives on the final expression. The standard lowering
pass also performs this expansion.

Differentiate the Lagrangian to obtain tangent, adjoint and Hessian forms.
Check the reduced gradient and Hessian action against JAX automatic differentiation.
"""

# %%
import basix
import jax
import numpy as np

import uflx_jax_compiler
from uflx.arithmetic import num
from uflx.calculus import directional_derivative, gradient, integral
from uflx.forms import apply, form, function
from uflx_basix.cells import cell
from uflx_basix.spaces import local_space

interval = cell(basix.CellType.interval)

P2 = basix.create_element(basix.ElementFamily.P, interval.cell_type, 2)
P1 = basix.create_element(basix.ElementFamily.P, interval.cell_type, 1)

state_space = local_space(interval, P2)
control_space = local_space(interval, P1)

m = function(control_space, "m")  # Reaction coefficient (control)
v = function(state_space, "v")  # Test function (state)
u = function(state_space, "u")  # Forward solution (state)

# %% [markdown]
# The state satisfies the PDE when its residual `F` vanishes for every test function `v`.

# %%
diffusion = gradient(u, interval) @ gradient(v, interval)
state_integral = integral(diffusion + m * u * v - v, interval)
F = form(state_integral, (v,))
J = integral((u - num(1)) ** num(2) / num(2), interval)

h, k = function(control_space, "h"), function(control_space, "k")  # Fixed control directions

u_h = function(state_space, "u_h")  # Tangent linear solution for h
tangent = directional_derivative(F, (u, m), (u_h, h))

# The same tangent equation defines u_k for direction k. Differentiating the
# equation for u_h along k gives the mixed tangent, with h held fixed.
u_k = function(state_space, "u_k")  # Tangent linear solution for k
u_hk = function(state_space, "u_hk")  # Mixed second-order tangent linear solution for h and k
second_tangent = directional_derivative(tangent, (u, m, u_h), (u_k, k, u_hk))

# %% [markdown]
# Choose the adjoint `p` so that varying the state makes no first-order change to the Lagrangian.

# %%
p = function(state_space, "p")  # Adjoint solution
L = J - apply(F, (p,))
adjoint = form(directional_derivative(L, u, v), (v,))

# At the state and adjoint solutions, varying m alone gives the reduced gradient.
reduced_gradient = form(directional_derivative(L, m, k), (k,))

# Both derivatives follow a control perturbation along h, including its effect on u and p.
p_h = function(state_space, "p_h")  # Incremental adjoint solution for h
fields = (u, m, p)
variation_h = (u_h, h, p_h)
incremental_adjoint = directional_derivative(adjoint, fields, variation_h)
hessian_action = directional_derivative(reduced_gradient, fields, variation_h)  # Linear form in k

# Degree 5 integrates a P1 coefficient times two P2 fields exactly.
rule = basix.make_quadrature(interval.cell_type, degree=5)
# The inputs after the expression identify each quadrature rule and coefficient field.
objective_kernel = jax.jit(uflx_jax_compiler.trace(J, J, u))
residual_kernel = jax.jit(uflx_jax_compiler.trace(F, state_integral, u, m))
gradient_kernel = jax.jit(uflx_jax_compiler.trace(reduced_gradient, J, state_integral, *fields))
hessian_kernel = jax.jit(
    uflx_jax_compiler.trace(hessian_action, J, state_integral, *fields, *variation_h)
)

# A constant control m0 gives constant state and adjoint solutions. For h = 1
# their tangents follow by differentiating those values with respect to m0.
m0 = 0.5
u_values = np.full(state_space.num_dofs, 1 / m0)
p_values = (u_values - 1) / m0
u_h_values = -u_values / m0
p_h_values = (u_h_values - p_values) / m0

m_values = np.full(control_space.num_dofs, m0)
h_values = np.ones(control_space.num_dofs)
values = (u_values, m_values, p_values)
variations = (u_h_values, h_values, p_h_values)
print("Objective", objective_kernel(rule, u_values))
# Each vector entry tests the form against one of the two P1 basis functions.
gradient_values = gradient_kernel(rule, rule, *values)
hessian_values = hessian_kernel(rule, rule, *values, *variations)
print("Reduced gradient", gradient_values)
print("Reduced Hessian action", hessian_values)


def lagrangian_value(state_dofs, control_dofs, adjoint_dofs):
    residual = residual_kernel(rule, state_dofs, control_dofs)
    return objective_kernel(rule, state_dofs) - adjoint_dofs @ residual


# At the state and adjoint solutions, the partial derivative with respect to
# control_dofs gives one reduced-gradient entry per P1 coefficient.
control_gradient = jax.grad(lagrangian_value, argnums=1)
# Follow the state, control and adjoint variations along h to get the Hessian action.
ad_gradient, ad_hessian_action = jax.jvp(control_gradient, values, variations)
np.testing.assert_allclose(gradient_values, ad_gradient, rtol=1e-6)
np.testing.assert_allclose(hessian_values, ad_hessian_action, rtol=1e-6)
print("JAX AD verified the reduced gradient and Hessian action for h = 1.")

# %% [markdown]
# ## Full Newton
#
# Treat the state, control and adjoint as independent unknowns.

# %%
# One product test argument collects the adjoint, control and state equations in that order.
q = function(state_space, "q")  # Adjoint test function
kkt_test = (v, k, q)
kkt_residual = form(directional_derivative(L, fields, kkt_test), (kkt_test,))

newton_step = (
    function(state_space, "delta_u"),
    function(control_space, "delta_m"),
    function(state_space, "delta_p"),
)
kkt_jacobian = form(directional_derivative(kkt_residual, fields, newton_step), (newton_step,))

kkt_residual_kernel = jax.jit(uflx_jax_compiler.trace(kkt_residual, J, state_integral, *fields))
kkt_jacobian_kernel = jax.jit(uflx_jax_compiler.trace(kkt_jacobian, J, state_integral, *fields))

# The Newton equation is K delta = -R. Both axes use P2 state, P1 control, P2 adjoint order.
newton_matrix = kkt_jacobian_kernel(rule, rule, *values)
newton_rhs = -kkt_residual_kernel(rule, rule, *values)
with np.printoptions(precision=4, suppress=True):
    print("Newton matrix (state, control, adjoint)", newton_matrix, sep="\n")
    print("Newton right-hand side", newton_rhs)

# %% [markdown]
# ## Gauss–Newton
#
# Linearise the tracking error and the PDE before forming the step.

# %%
# The tracking error u - 1 varies by delta_u. Its least-squares term gives the state mass block.
delta_u, delta_m, delta_p = newton_step
tracking_mass = integral(delta_u * v, interval)
pde_step = directional_derivative(F, (u, m), (delta_u, delta_m))
pde_test = directional_derivative(F, (u, m), (v, k))
gauss_newton = form(
    tracking_mass - apply(pde_step, (q,)) - apply(pde_test, (delta_p,)),
    (kkt_test, newton_step),
)
gauss_newton_kernel = jax.jit(
    uflx_jax_compiler.trace(gauss_newton, tracking_mass, state_integral, u, m)
)
gauss_newton_matrix = gauss_newton_kernel(rule, rule, u_values, m_values)

# The step uses newton_rhs. Only the two blocks from differentiating m * u twice drop out.
with np.printoptions(precision=4, suppress=True):
    print("Gauss-Newton matrix (state, control, adjoint)", gauss_newton_matrix, sep="\n")

# For h = 1 the Gauss-Newton incremental adjoint has constant source u_h,
# so its solution is u_h / m0. The control rows then give the reduced action.
gn_p_h_values = u_h_values / m0
gn_step = np.concatenate((u_h_values, h_values, gn_p_h_values))
control_rows = slice(state_space.num_dofs, state_space.num_dofs + control_space.num_dofs)
gn_hessian_values = (gauss_newton_matrix @ gn_step)[control_rows]
print("Reduced Gauss-Newton action", gn_hessian_values)
