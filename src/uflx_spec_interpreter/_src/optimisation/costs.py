# SPDX-License-Identifier: MIT
"""Generic extraction costs; opaque operations have no estimated internal work."""

from egglog import BaseExpr, EGraph
from egglog.deconstruct import get_callable_args

from .language import Scalar


def loop_cost(
    egraph: EGraph, expr: BaseExpr, children_costs: list[tuple[int, int]]
) -> tuple[int, int]:
    """Estimate arithmetic iterations, using tree size to break ties.

    Opaque atoms cost zero; this is a loop-sensitive proxy, not a hardware model.
    """
    flops = sum(c[0] for c in children_costs)
    size = 1 + sum(c[1] for c in children_costs)
    if (args := get_callable_args(expr, Scalar.summed)) is not None:
        extent = args[2].value
        # Positive weighting keeps extraction monotone, even for empty dimensions.
        flops = max(1, extent) * children_costs[0][0] + max(0, extent - 1)
    elif (
        get_callable_args(expr, Scalar.__add__) is not None
        or get_callable_args(expr, Scalar.__mul__) is not None
    ):
        flops += 1
    elif get_callable_args(expr, Scalar.__truediv__) is not None:
        # Prefer multiplication by an exact reciprocal when both are available.
        flops += 2
    return flops, size
