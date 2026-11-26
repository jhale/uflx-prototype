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
from uflx.algorithms import occurs
from uflx.forms import form, function
from uflx_basix.spaces import real_scalar_space
from uflx_extensions.external_operators import (
    OutputType,
    external_operator,
    jvp,
    operation,
    vjp,
)


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
