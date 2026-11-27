# Copyright (C) 2026 Jack S. Hale
#
# This file is part of uflx-core-experimental (https://fenics.github.io/uflx/language/)
#
# SPDX-License-Identifier:    MIT
"""Supply whole fields or their point values to JAX external functions."""

from collections.abc import Callable
from dataclasses import dataclass

import jax.numpy as jnp

from uflx_extensions.external_operators import PullbackComponent

from ._context import as_argument
from ._forms import supplied


@dataclass(frozen=True)
class Pointwise:
    """Pass operand values at the current point to a JAX-traceable function."""

    function: Callable


def _call_external(node, binding, site, value):
    if isinstance(binding, Pointwise):
        return binding.function(*(value(x) for x in node.operands))

    operands = (supplied(x, x, site, value) for x in node.operands)
    return binding(*(as_argument(x, site) for x in operands))


def evaluate_external(node, bindings, site, value, memo):
    op = node.operation
    if isinstance(op, PullbackComponent):
        key = (op.pullback, node.operands)
        if key not in memo:
            result = _call_external(node, bindings[op.pullback], site, value)
            count = len(node.operands) - 1  # The final operand is the output seed.
            if not isinstance(result, tuple) or len(result) != count:
                raise ValueError(f"external pullback must return a tuple of {count} cotangents")
            memo[key] = result
        result = memo[key][op.operand]
    else:
        result = _call_external(node, bindings[op], site, value)
    if callable(result) and node.domain is not None:
        result = result(site.points[node.domain])
    return jnp.asarray(result)
