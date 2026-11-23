"""Supply whole fields or their point values to JAX external functions."""

from collections.abc import Callable
from dataclasses import dataclass

import jax.numpy as jnp

from ._context import as_argument
from ._forms import supplied


@dataclass(frozen=True)
class Pointwise:
    """Pass operand values at the current point to a JAX-traceable function."""

    function: Callable


def evaluate_external(node, binding, site, value):
    if isinstance(binding, Pointwise):
        return jnp.asarray(binding.function(*(value(x) for x in node.operands)))

    operands = (supplied(x, x, site, value) for x in node.operands)
    result = binding(*(as_argument(x, site) for x in operands))
    if callable(result) and node.domain is not None:
        result = result(site.points[node.domain])
    return jnp.asarray(result)
