# SPDX-License-Identifier: MIT
"""Conservative facts used to guard removal of potentially undefined expressions."""

from collections.abc import Callable
from functools import cache

from ..nodes import (
    Absolute,
    AbstractExpression,
    BroadcastTo,
    Complex,
    ComponentTensor,
    Conjugate,
    Division,
    Float,
    ImaginaryPart,
    Index,
    Indexed,
    IndexSum,
    Integer,
    Product,
    RealPart,
    Sum,
    Tensor,
    Zero,
)


def definedness() -> Callable[[AbstractExpression], bool]:
    """Return a per-invocation fact cache; False means unknown, not undefined.

    Literal arithmetic and structurally total operations are recognised. Runtime
    terminals and extension nodes are deliberately unknown. These facts concern
    mathematical values, not floating-point overflow or NaNs.
    """

    @cache
    def known(expression: AbstractExpression) -> bool:
        match expression:
            case Integer() | Float() | Complex() | Zero():
                return True
            case Sum() | Product():
                return known(expression.first) and known(expression.second)
            case Division():
                denominator = expression.second
                return (
                    isinstance(denominator, Integer | Float)
                    and denominator.value != 0
                    and known(expression.first)
                )
            case Absolute() | Conjugate() | RealPart() | ImaginaryPart():
                return known(expression.operand)
            case Tensor():
                return all(known(entry) for entry in expression.entries)
            case BroadcastTo():
                return known(expression.value)
            case ComponentTensor():
                return known(expression.expression)
            case Indexed():
                return all(isinstance(i, int | Index) for i in expression.indices) and known(
                    expression.expression
                )
            case IndexSum():
                extents = dict(expression.expression.free_indices)
                return any(extents[i] == 0 for i in expression.indices) or known(
                    expression.expression
                )
            case _:
                return False

    return known
