"""Type compatibility and completion of expression rewrites."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from .nodes import AbstractExpression, Zero
from .static import _free, _free_indices, _shape
from .traversal import _map_values

if TYPE_CHECKING:
    from .nodes.spaces import AbstractFunctionSpace
__all__ = ["_complete_indices", "_promote", "_rewrite", "_zero_like"]


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
    is_zero = isinstance(replacement, Zero)
    replacement = _promote(original, replacement)
    if is_zero and (actual.keys() != expected.keys() or not isinstance(replacement, Zero)):
        return _zero_like(original, original.argument_spaces)
    if actual.keys() == expected.keys():
        return replacement
    from .names import _derived_indices
    from .tensors import broadcast_to, component_tensor, indexed

    # SH06. Selecting a repeated value preserves an index without introducing a sum.
    for i, n in original.free_indices:
        if i not in actual:
            axes = _derived_indices(
                "uflx.complete_indices", (original, replacement, i), len(replacement.value_shape)
            )
            replacement = component_tensor(
                indexed(broadcast_to(replacement, (n, *replacement.value_shape)), (i, *axes)),
                axes,
            )
    return replacement


def _promote(original: AbstractExpression, replacement: AbstractExpression) -> AbstractExpression:
    from .arithmetic import multiply
    from .nodes import Complex, Integer, ScalarType

    if original.scalar_type == ScalarType.complex and replacement.scalar_type == ScalarType.real:
        return multiply(Complex(Integer(1), Integer(0)), replacement)
    if original.scalar_type != replacement.scalar_type:
        raise ValueError("a rewrite changes scalar type")
    return replacement


def _rewrite(
    f: AbstractExpression, rule: Callable[[AbstractExpression], AbstractExpression]
) -> AbstractExpression:
    """SH19-SH21. Visit operands, then complete and visit replacements.

    Returning the node retains it for this visit. The memo and cycle check belong
    to this invocation; they add no state to expression nodes.
    """
    done: dict[AbstractExpression, AbstractExpression] = {}
    active: set[AbstractExpression] = set()

    def visit(e: AbstractExpression) -> AbstractExpression:
        if e in done:
            return done[e]
        if e in active:
            raise ValueError("rewrite cycle", e)
        active.add(e)
        rebuilt = _map_values(e, visit)
        replacement = rule(rebuilt)
        if replacement != rebuilt:
            replacement = _complete_indices(rebuilt, replacement)
        out = rebuilt if replacement == rebuilt else visit(replacement)
        active.remove(e)
        done[e] = out
        return out

    return visit(f)
