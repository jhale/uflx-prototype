"""Type compatibility and completion of expression rewrites."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .nodes import AbstractExpression, Zero
from .static import _free, _free_indices, _shape

if TYPE_CHECKING:
    from .nodes.spaces import AbstractFunctionSpace
__all__ = ["_complete_indices", "_promote", "_zero_like"]


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
