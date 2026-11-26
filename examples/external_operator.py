# Copyright (C) 2026 Jack S. Hale
#
# This file is part of uflx-core-experimental (https://fenics.github.io/uflx/language/)
#
# SPDX-License-Identifier:    MIT

# %% [markdown]
"""# A scalar external operator

Differentiate the square in UFLx and compare its tangent action with JAX AD.
"""

# %%
from functools import partial

import jax
import jax.numpy as jnp

from uflx.calculus import directional_derivative
from uflx.forms import function
from uflx_basix.spaces import real_scalar_space
from uflx_extensions.external_operators import (
    AbstractOperation,
    OutputType,
    apply_external_derivatives,
    external_operator,
    operation,
)
from uflx_jax_compiler import trace


def scalar_result(operands):
    if any(x.value_shape or x.domain is not None or x.scalar_type != "real" for x in operands):
        raise ValueError("expected real scalar inputs in the global scalar space")
    return OutputType((), "real", None)


square_tangent = operation("example.square_tangent", 2, scalar_result)


class Square(AbstractOperation):
    identifier = "example.Square"

    def output_type(self, operands):
        (x,) = operands
        return scalar_result((x,))

    def derivative_action(self):
        return square_tangent


square = Square()
V = real_scalar_space()
x, h = function(V, "x"), function(V, "h")
expression = external_operator(square, (x,))
f = partial(trace(expression, square, x), jnp.square)

# UFLx expands D square(x)[h] to the external action square_tangent(x, h).
derivative = apply_external_derivatives(directional_derivative(expression, x, h))
assert derivative is external_operator(square_tangent, (x, h))


def tangent(x, h):
    return 2 * x * h


df = partial(trace(derivative, square_tangent, x, h), tangent)
value, jax_tangent = jax.jvp(f, (3.0,), (2.0,))
uflx_tangent = df(3.0, 2.0)
assert value == 9.0
assert uflx_tangent == jax_tangent == 12.0
print("square(3) =", value, "  D square(3)[2] =", uflx_tangent)
