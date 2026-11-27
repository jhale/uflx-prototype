# Copyright (C) 2026 Jack S. Hale
#
# This file is part of uflx-core-experimental (https://fenics.github.io/uflx/language/)
#
# SPDX-License-Identifier:    MIT
"""Numerical oracle for checking symbolic lowering."""

from uflx.nodes import AbstractExpression
from uflx_jax_compiler.interpreter import _check_inputs, _kernel

from . import _autodiff


def reference_trace(expression: AbstractExpression, *inputs):
    """Return a JAX-traceable reference function for an expression or form.

    Evaluate unlowered derivatives with JAX AD to check symbolic lowering.
    Inputs and form slots follow the same convention as trace.
    """
    _check_inputs(expression, inputs)
    kernel = _kernel(expression, inputs, reference=_autodiff)
    kernel.inputs = inputs
    return kernel
