# SPDX-License-Identifier: MIT
"""Guarded delta contraction using UFLx's capture-avoiding index substitution."""

from collections.abc import Callable, Iterator

from ..algorithms.rewriting import complete_indices
from ..algorithms.substitution import occurs_in_metadata, substitute_values
from ..nodes import AbstractExpression, Index, Indexed, IndexSum, Integer, Product, Tensor
from ..tensors import identity, index_sum


def _delta_factors(
    expression: AbstractExpression,
) -> Iterator[tuple[Index, Index, int, AbstractExpression]]:
    """Find identity entries in a product, retaining the other factors' grouping."""
    match expression:
        case Indexed(expression=Tensor() as source, indices=(Index() as i, Index() as j)) if i != j:
            n = source.value_shape[0]
            if isinstance(n, int) and source.value_shape == (n, n) and source == identity(n):
                yield i, j, n, Integer(1)
        case Product(first=a, second=b):
            for i, j, n, rest in _delta_factors(a):
                yield i, j, n, rest * b
            for i, j, n, rest in _delta_factors(b):
                yield i, j, n, a * rest
        case _:
            return


def contract_delta(
    expression: AbstractExpression, known_defined: Callable[[AbstractExpression], bool]
) -> AbstractExpression:
    """Remove one delta-bound index without discarding unknown singularities."""
    if not isinstance(expression, IndexSum):
        return expression
    body, bound = expression.expression, expression.indices
    extents = dict(body.free_indices)
    for i, j, extent, rest in _delta_factors(body):
        for target, removed in ((i, j), (j, i)):
            if (
                removed not in bound
                or extents[removed] != extent
                or extents[target] != extent
                or not known_defined(rest)
                or occurs_in_metadata(removed, rest)
            ):
                continue
            substitution = {removed: target}
            selected = substitute_values(rest, substitution)
            # A delta may be the only occurrence of an index which must survive.
            template = substitute_values(body, substitution)
            selected = complete_indices(template, selected)
            return index_sum(selected, tuple(index for index in bound if index != removed))
    return expression
