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

from uflx.forms import apply, form, function
from uflx_basix.spaces import real_scalar_space
from uflx_extensions.external_operators import (
    external_operator,
    jvp,
    operation,
    output_type,
    vjp,
)
from uflx_jax_compiler import emit_c, trace


def square_type(operands):
    (x,) = operands
    return output_type(x.value_shape, x.scalar_type, x.domain)


square = operation("example.square", 1, square_type)
V = real_scalar_space()
u, v, p, h, z = (function(V, name) for name in ("u", "v", "p", "h", "z"))
square_u = external_operator(square, (u,))
R = form(square_u * v, (v,))

Jh = jvp(R, u, p, h)  # Supply a perturbation; the test slot stays open.
JTz = vjp(R, u, p, z)  # Supply a test value; the perturbation slot stays open.

pushforward = square.jvp_operator()
pullback = square.vjp_operator()

# %% [markdown]
# ## Bind both derivative actions to the same numerical square


# %%
def square_jvp(x, dx):
    _, tangent = jax.jvp(jnp.square, (x,), (dx,))
    return tangent


def square_vjp(x, seed):
    _, back = jax.vjp(jnp.square, x)
    return back(seed)  # One cotangent per operand, in order.


forward_pairing = apply(Jh, (z,))
reverse_pairing = apply(JTz, (h,))

evaluate_forward = trace(forward_pairing, square, pushforward, p, h, z)
evaluate_reverse = trace(reverse_pairing, pullback, p, h, z)
bound_forward = partial(evaluate_forward, jnp.square, square_jvp)
bound_reverse = partial(evaluate_reverse, square_vjp)

tangent = jax.jit(bound_forward)
inputs = (3.0, 2.0, 4.0)  # p, h, z
value = tangent(*inputs)
assert value == bound_reverse(*inputs) == 48.0
print(value)

# %% [markdown]
# ## Emit the tangent as C99

# %%
scalar = jax.ShapeDtypeStruct((), jnp.float32)
export_tangent = export.export(tangent)
exported = export_tangent(scalar, scalar, scalar)  # p, h, z
tangent_c = emit_c(exported, "square_tangent")
print(tangent_c)
