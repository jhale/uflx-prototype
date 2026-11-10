# SPDX-License-Identifier: MIT
"""Small structural rewrites that do not rearrange scalar arithmetic."""

from collections.abc import Callable

from ..algorithms.rewriting import zero_like
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
    expression: AbstractExpression, known_defined: Callable[[AbstractExpression], bool]
) -> AbstractExpression:
    """Select stored components and remove redundant complex operations.

    General binder substitution and gathered indices are left untouched. The
    caller completes static metadata and visits any replacement.
    """
    match expression:
        case Conjugate() if isinstance(expression.operand, Conjugate):
            return expression.operand.operand
        case RealPart() if expression.operand.scalar_type == "real":
            return expression.operand
        case ImaginaryPart() if expression.operand.scalar_type == "real" and known_defined(
            expression.operand
        ):
            return zero_like(expression, expression.argument_spaces)
        case IndexSum():
            extents = dict(expression.expression.free_indices)
            if any(extents[i] == 0 for i in expression.indices):
                return zero_like(expression, expression.argument_spaces)
        case Indexed() if all(isinstance(i, int | Index) for i in expression.indices):
            source = expression.expression
            if isinstance(source, Zero):
                return zero_like(expression, expression.argument_spaces)
            if isinstance(source, BroadcastTo) and not source.value.value_shape:
                return source.value
            if (
                isinstance(source, Tensor)
                and isinstance(first := expression.indices[0], int)
                and known_defined(source)
            ):
                return indexed(source.entries[first], expression.indices[1:])
        case Tensor() if all(
            isinstance(e, Zero) or (isinstance(e, Integer) and e.value == 0)
            for e in expression.entries
        ):
            return zero_like(expression, expression.argument_spaces)
        case _:
            pass
    return expression
