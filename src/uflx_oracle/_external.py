"""Supply whole fields or their point values to JAX external functions."""

from collections.abc import Callable
from dataclasses import dataclass

import jax.numpy as jnp

from ._context import call
from ._forms import supplied


@dataclass(frozen=True)
class Pointwise:
    """Pass operand values at the current point to a JAX-traceable function."""

    function: Callable


def evaluate_external(node, frame):
    binding = frame.environment[node.operation]
    if isinstance(binding, Pointwise):
        return jnp.asarray(binding.function(*(frame.value(x) for x in node.operands)))

    result = call(frame.site, binding, *(supplied(x, x, frame) for x in node.operands))
    if callable(result) and node.domain is not None:
        result = result(frame.site.points[node.domain])
    return jnp.asarray(result)
