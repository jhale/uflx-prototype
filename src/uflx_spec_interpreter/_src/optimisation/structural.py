# SPDX-License-Identifier: MIT
"""Small structural rewrites that do not rearrange scalar arithmetic."""

from collections.abc import Callable

from ..algorithms.rewriting import complete_indices, zero_like
from ..nodes import (
    AbstractExpression,
    BroadcastTo,
    Conjugate,
    ImaginaryPart,
    Index,
    Indexed,
    IndexSum,
    Integer,
    RealPart,
    Tensor,
    Zero,
)
from ..tensors import indexed


def simplify(
    expression: AbstractExpression,
    known_defined: Callable[[AbstractExpression], bool],
    *,
    preserve_evaluation: bool,
) -> AbstractExpression:
    """Apply one structural rule and restore compatible static metadata."""
    replacement = _simplify(expression, known_defined)
    if replacement == expression or (
        preserve_evaluation and replacement.scalar_type != expression.scalar_type
    ):
        return expression
    return complete_indices(expression, replacement)


def _simplify(
    expression: AbstractExpression, known_defined: Callable[[AbstractExpression], bool]
) -> AbstractExpression:
    """Select stored components and remove redundant complex operations.

    General binder substitution and gathered indices are left untouched. The
    caller completes static metadata and visits any replacement.
    """
    match expression:
        case Conjugate(operand=Conjugate(operand=a)):
            return a
        case RealPart(operand=a) if a.scalar_type == "real":
            return a
        case ImaginaryPart(operand=a) if a.scalar_type == "real" and known_defined(a):
            return zero_like(expression, expression.argument_spaces)
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
