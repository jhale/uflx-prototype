# SPDX-License-Identifier: MIT
"""Exact evaluation of small stored matrices, without floating-point arithmetic."""

from collections.abc import Callable
from fractions import Fraction

from ..algorithms.substitution import substitute_values
from ..literals import num
from ..nodes import (
    AbstractExpression,
    BroadcastTo,
    ComponentTensor,
    Determinant,
    Division,
    GatheredIndex,
    Index,
    Indexed,
    Integer,
    Inverse,
    Tensor,
    Zero,
)


def _entry(
    expression: AbstractExpression, indices: tuple[int | Index | GatheredIndex, ...]
) -> Fraction | None:
    """Read a stored integer/rational entry; unknown or floating values stay opaque."""
    match expression:
        case Zero():
            return Fraction(0)
        case Tensor(entries=entries) if indices and isinstance(indices[0], int):
            return _entry(entries[indices[0]], indices[1:])
        case BroadcastTo(value=value):
            leading = len(expression.value_shape) - len(value.value_shape)
            selected = tuple(
                0 if extent == 1 else i
                for i, extent in zip(indices[leading:], value.value_shape, strict=True)
            )
            return _entry(value, selected)
        case Indexed(expression=source, indices=selected) if all(
            isinstance(i, int | Index) for i in selected
        ):
            return _entry(source, (*selected, *indices))
        case ComponentTensor(expression=body, indices=bound) if len(indices) == len(bound):
            return _entry(substitute_values(body, dict(zip(bound, indices, strict=True))), ())
        case Integer(value=value) if not indices:
            return Fraction(value)
        case Division(first=Integer(value=a), second=Integer(value=b)) if not indices and b:
            return Fraction(a, b)
        case _:
            return None


def _minor(
    matrix: tuple[tuple[Fraction, ...], ...], row: int, column: int
) -> tuple[tuple[Fraction, ...], ...]:
    return tuple(
        tuple(v for j, v in enumerate(values) if j != column)
        for i, values in enumerate(matrix)
        if i != row
    )


def _determinant(matrix: tuple[tuple[Fraction, ...], ...]) -> Fraction:
    if not matrix:
        return Fraction(1)
    return sum(
        (
            (-1) ** j * value * _determinant(_minor(matrix, 0, j))
            for j, value in enumerate(matrix[0])
        ),
        Fraction(0),
    )


def _literal(value: Fraction) -> AbstractExpression:
    numerator = num(value.numerator)
    return numerator if value.denominator == 1 else numerator / num(value.denominator)


def fold_matrix(
    expression: AbstractExpression, known_defined: Callable[[AbstractExpression], bool]
) -> AbstractExpression:
    """Fold rational determinants and nonsingular square inverses up to 3 by 3.

    The small size limit bounds cofactor expansion. Singular inverses and other
    matrices retain their original operation. Call only when changing floating-
    point evaluation is allowed; the caller completes type and family metadata.
    """
    if not isinstance(expression, Determinant | Inverse):
        return expression
    source = expression.operand
    rows, columns = source.value_shape
    if rows != columns or not isinstance(rows, int) or rows > 3:
        return expression
    if not known_defined(source):
        return expression
    entries = tuple(tuple(_entry(source, (i, j)) for j in range(rows)) for i in range(rows))
    if any(value is None for row in entries for value in row):
        return expression
    # Filter the checked optionals to retain precise types without casts.
    matrix = tuple(tuple(value for value in row if value is not None) for row in entries)
    determinant = _determinant(matrix)
    if isinstance(expression, Determinant):
        return _literal(determinant)
    if not determinant:
        return expression
    if rows == 0:
        return source
    return Tensor(
        tuple(
            Tensor(
                tuple(
                    _literal((-1) ** (i + j) * _determinant(_minor(matrix, j, i)) / determinant)
                    for j in range(rows)
                )
            )
            for i in range(rows)
        )
    )
