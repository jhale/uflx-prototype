"""Expand tensor arithmetic into scalar components after spatial lowering."""

from __future__ import annotations

from ..algorithms.names import derived_indices
from ..algorithms.rewriting import rewrite
from ..nodes import AbstractExpression, ComponentTensor, Division, Indexed, Product, Sum


def expand_arithmetic(f: AbstractExpression) -> AbstractExpression:
    """ST13-ST16. Expand tensor sums, scalar products and quotients."""

    def expand(e: AbstractExpression) -> AbstractExpression:
        if not isinstance(e, Sum | Product | Division) or not e.value_shape:
            return e
        indices = derived_indices("uflx.expand_arithmetic", (e,), len(e.value_shape))
        first, second = (
            (e.numerator, e.denominator) if isinstance(e, Division) else (e.first, e.second)
        )
        if first.value_shape:
            first = Indexed(first, indices)
        if second.value_shape:
            second = Indexed(second, indices)
        scalar = type(e)(first, second)
        return ComponentTensor(scalar, indices)

    return rewrite(f, expand)
