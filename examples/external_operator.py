# Copyright (C) 2026 Jack S. Hale
#
# This file is part of uflx-core-experimental (https://fenics.github.io/uflx/language/)
#
# SPDX-License-Identifier:    MIT

# %% [markdown]
"""# A scalar external operator

A residual with one external law. Forward and reverse differentiation create
symbolic calls to the law's tangent and pullback, respectively.
"""

# %%
from functools import partial

import jax
import jax.numpy as jnp
from jax import export

from uflx.algorithms import occurs
from uflx.forms import apply, form, function
from uflx_basix.spaces import real_scalar_space
from uflx_extensions.external_operators import (
    OutputType,
    external_operator,
    jvp,
    operation,
    vjp,
)
from uflx_jax_compiler import emit_c, trace


def square_type(operands):
    (x,) = operands
    return OutputType(x.value_shape, x.scalar_type, x.domain)


square = operation("example.square", 1, square_type)
V = real_scalar_space()
u, v, h, z = (function(V, name) for name in ("u", "v", "h", "z"))
R = form(external_operator(square, (u,)) * v, (v,))

Jh = jvp(R, u, h)  # Supply a perturbation; the test slot stays open.
JTz = vjp(R, u, z)  # Supply a test value; the perturbation slot stays open.

assert occurs(square.derivative_action(), Jh)
assert occurs(square.vjp_action(0), JTz)

# %% [markdown]
# ## Evaluate the tangent with JAX


# %%
def square_jvp(x, dx):
    _, tangent = jax.jvp(jnp.square, (x,), (dx,))
    return tangent


# Bind the primal and its JAX-derived tangent, then evaluate the form at z.
evaluate = trace(apply(Jh, (z,)), square, square.derivative_action(), u, h, z)
tangent = jax.jit(partial(evaluate, jnp.square, square_jvp))
value = tangent(3.0, 2.0, 4.0)
assert value == 48.0  # D square(3)[2] * 4
print(value)

# %% [markdown]
# ## Emit the tangent as C99

# %%
scalar = jax.ShapeDtypeStruct((), jnp.float32)
exported = export.export(tangent)(scalar, scalar, scalar)  # u, h, z
tangent_c = emit_c(exported, "square_tangent")
print(tangent_c)
