# SPDX-License-Identifier: MIT
"""Conservative facts used to guard removal of potentially undefined expressions."""

from collections.abc import Callable

from ..nodes import (
    Absolute,
    AbstractExpression,
    BroadcastTo,
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
)


def is_defined(known: Callable[[AbstractExpression], bool], expression: AbstractExpression) -> bool:
    """Check known mathematical definedness; False means unknown, not undefined.

    Compose with the core's memoized traversal to share facts within one run.
    Terminals denote supplied, defined values on admissible inputs. Nonterminal
    extensions remain unknown. Floating-point overflow and NaNs are outside this
    mathematical check.
    """
    if expression.is_terminal:
        return True
    match expression:
        case Sum(first=a, second=b) | Product(first=a, second=b):
            return known(a) and known(b)
        case Division(first=a, second=Integer(value=value) | Float(value=value)):
            return value != 0 and known(a)
        case (
            Absolute(operand=a)
            | Conjugate(operand=a)
            | RealPart(operand=a)
            | ImaginaryPart(operand=a)
        ):
            return known(a)
        case Tensor(entries=entries):
            return all(map(known, entries))
        case BroadcastTo(value=a) | ComponentTensor(expression=a):
            return known(a)
        case Indexed(expression=a, indices=indices):
            return all(isinstance(i, int | Index) for i in indices) and known(a)
        case IndexSum(expression=a, indices=indices):
            extents = dict(a.free_indices)
            return any(extents[i] == 0 for i in indices) or known(a)
        case _:
            return False
