# SPDX-License-Identifier: MIT
"""Small structural rewrites that do not rearrange scalar arithmetic."""

from collections.abc import Callable

from ..algorithms.rewriting import complete_indices, zero_like
from ..algorithms.substitution import occurs_in_metadata, substitute_values
from ..nodes import (
    AbstractExpression,
    BroadcastTo,
    ComponentTensor,
    Index,
    Indexed,
    IndexSum,
    Integer,
    Tensor,
    Zero,
)
from ..tensors import indexed
from .constants import fold_matrix


def simplify(
    expression: AbstractExpression,
    known_defined: Callable[[AbstractExpression], bool],
    *,
    preserve_evaluation: bool,
) -> AbstractExpression:
    """Apply one structural rule and restore compatible static metadata."""
    replacement = expression if preserve_evaluation else fold_matrix(expression, known_defined)
    if replacement == expression:
        replacement = _simplify(expression, known_defined)
    if replacement == expression or (
        preserve_evaluation and replacement.scalar_type != expression.scalar_type
    ):
        return expression
    return complete_indices(expression, replacement)


def _simplify(
    expression: AbstractExpression, known_defined: Callable[[AbstractExpression], bool]
) -> AbstractExpression:
    """Select stored components and simplify empty sums.

    Component selection uses capture-avoiding value substitution. Gathered
    indices remain untouched. The caller completes static metadata.
    """
    match expression:
        case IndexSum(expression=a, indices=indices):
            extents = dict(a.free_indices)
            if any(extents[i] == 0 for i in indices):
                return zero_like(expression, expression.argument_spaces)
        case Indexed(expression=source, indices=indices) if all(
            isinstance(i, int | Index) for i in indices
        ):
            match source:
                case Zero():
                    return zero_like(expression, expression.argument_spaces)
                case BroadcastTo(value=a) if not a.value_shape:
                    return a
                case ComponentTensor(expression=body, indices=bound) if not any(
                    occurs_in_metadata(i, body) for i in bound
                ) and (
                    known_defined(body)
                    or (
                        isinstance(body, Indexed)
                        and len(body.indices) == len(bound)
                        and set(body.indices) == set(bound)
                        and not set(bound).intersection(dict(body.expression.free_indices))
                    )
                ):
                    return substitute_values(body, dict(zip(bound, indices, strict=True)))
                case Tensor(entries=entries) if isinstance(indices[0], int) and known_defined(
                    source
                ):
                    return indexed(entries[indices[0]], indices[1:])
                case _:
                    pass
        case Tensor(entries=entries) if all(
            isinstance(e, Zero) or (isinstance(e, Integer) and e.value == 0) for e in entries
        ):
            return zero_like(expression, expression.argument_spaces)
        case _:
            pass
    return expression
