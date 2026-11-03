from __future__ import annotations

from typing import TYPE_CHECKING

from .nodes import Complex, Float, Integer, Zero

if TYPE_CHECKING:
    from .nodes.base import AbstractExpression, Base
    from .nodes.domains import AbstractDomain
    from .nodes.indices import AbstractExtent, Index
    from .nodes.spaces import AbstractFunctionSpace
    from .nodes.topology import AbstractTopology


# The free functions read the static type of an expression.
# Shared implementation helpers; the package public API is declared in __init__.py.
__all__ = [
    "_arity",
    "_complete_indices",
    "_field_terminal",
    "_free",
    "_free_indices",
    "_promote",
    "_shape",
    "_support",
    "_zero_like",
]


def _shape(e: AbstractExpression) -> tuple[int | AbstractExtent, ...]:
    return e.value_shape


def _free(e: AbstractExpression) -> dict[Index, int | AbstractExtent]:
    return dict(e.free_indices)


def _arity(e: AbstractExpression) -> int:
    return len(e.argument_spaces)


def _support(e: AbstractExpression) -> AbstractDomain | AbstractTopology | None:
    return e.domain


def _free_indices(*es: AbstractExpression) -> tuple[tuple[Index, int | AbstractExtent], ...]:
    merged: dict[Index, int | AbstractExtent] = {}
    for e in es:
        merged |= _free(e)
    return tuple(merged.items())


def _zero_like(
    f: AbstractExpression,
    argument_spaces: tuple[AbstractFunctionSpace | tuple[AbstractFunctionSpace, ...], ...],
    *others: AbstractExpression,
) -> Zero:
    return Zero(
        _shape(f),
        _free_indices(f, *others),
        argument_spaces,
        f.scalar_type,
    )


def _field_terminal(e: Base) -> bool:
    # Literal terminals have their own evaluation rows; fields may remain unevaluated.
    return e.is_terminal and not isinstance(e, (Integer, Float, Complex, Zero))


def _complete_indices(
    original: AbstractExpression, replacement: AbstractExpression
) -> AbstractExpression:
    """Restore a rewrite's lost family indices, rejecting incompatible signatures.

    Promote real replacements into the original complex arithmetic before
    restoring indices. Call before rebuilding a parent that binds an index.
    """
    if (
        _shape(original) != _shape(replacement)
        or original.argument_spaces != replacement.argument_spaces
    ):
        raise ValueError("index completion requires matching shapes and argument spaces")
    if replacement.domain not in (None, original.domain):
        raise ValueError("a replacement changes domain")
    expected, actual = _free(original), _free(replacement)
    if any(i not in expected or expected[i] != n for i, n in actual.items()):
        raise ValueError("a replacement introduces an index or changes its extent")
    replacement = _promote(original, replacement)
    if actual.keys() == expected.keys():
        return replacement
    from .arithmetic import add

    return add(replacement, _zero_like(original, original.argument_spaces))


def _promote(original: AbstractExpression, replacement: AbstractExpression) -> AbstractExpression:
    from .arithmetic import multiply
    from .nodes import Complex, Integer, ScalarType

    if original.scalar_type == ScalarType.complex and replacement.scalar_type == ScalarType.real:
        return multiply(Complex(Integer(1), Integer(0)), replacement)
    if original.scalar_type != replacement.scalar_type:
        raise ValueError("a rewrite changes scalar type")
    return replacement
