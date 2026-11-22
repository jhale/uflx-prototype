"""Type compatibility and completion of expression rewrites."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from ..nodes import AbstractExpression, Zero
from ..nodes.static import free_indices
from .traversal import map_values

if TYPE_CHECKING:
    from ..nodes.indices import AbstractExtent, Index
    from ..nodes.spaces import AbstractFunctionSpace


def zero_like(
    f: AbstractExpression,
    argument_spaces: tuple[AbstractFunctionSpace | tuple[AbstractFunctionSpace, ...], ...],
    *others: AbstractExpression,
) -> Zero:
    if f.scalar_type == "boolean":
        raise ValueError("Boolean zero is undefined")
    return Zero(
        f.value_shape,
        free_indices(f, *others),
        argument_spaces,
        f.scalar_type,
    )


def complete_indices(
    original: AbstractExpression, replacement: AbstractExpression
) -> AbstractExpression:
    """Restore a rewrite's lost family indices, rejecting incompatible signatures.

    Promote real replacements into the original complex arithmetic before
    restoring indices. Call before rebuilding a parent that binds an index.
    """
    if (
        original.value_shape != replacement.value_shape
        or original.argument_spaces != replacement.argument_spaces
    ):
        raise ValueError("index completion requires matching shapes and argument spaces")
    if replacement.domain not in (None, original.domain):
        raise ValueError("a replacement changes domain")
    return complete_family(original.free_indices, promote(original, replacement))


def complete_family(
    indices: tuple[tuple[Index, int | AbstractExtent], ...], replacement: AbstractExpression
) -> AbstractExpression:
    """Restore missing family axes after substitution, including unused arguments."""
    expected, actual = dict(indices), dict(replacement.free_indices)
    if not actual.items() <= expected.items():
        raise ValueError("a replacement introduces an index or changes its extent")
    if actual.keys() == expected.keys():
        return replacement
    if isinstance(replacement, Zero):
        return Zero(
            replacement.value_shape, indices, replacement.argument_spaces, replacement.scalar_type
        )
    from ..tensors import broadcast_to, component_tensor, indexed
    from .names import derived_indices

    # SH06. Selecting a repeated value preserves an index without introducing a sum.
    for i, n in indices:
        if i in actual:
            continue
        axes = derived_indices(
            "uflx.complete_indices", (indices, replacement, i), len(replacement.value_shape)
        )
        repeated = broadcast_to(replacement, (n, *replacement.value_shape))
        selected = indexed(repeated, (i, *axes))
        replacement = component_tensor(selected, axes)
    return replacement


def promote(original: AbstractExpression, replacement: AbstractExpression) -> AbstractExpression:
    from ..arithmetic import multiply
    from ..nodes import Complex, Integer, ScalarType

    if original.scalar_type == replacement.scalar_type:
        return replacement
    if original.scalar_type != ScalarType.complex or replacement.scalar_type != ScalarType.real:
        raise ValueError("a rewrite changes scalar type")
    if isinstance(replacement, Zero):
        return Zero(
            replacement.value_shape,
            replacement.free_indices,
            replacement.argument_spaces,
            ScalarType.complex,
        )
    return multiply(Complex(Integer(1), Integer(0)), replacement)


def rewrite(
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
        rebuilt = map_values(e, visit)
        replacement = rule(rebuilt)
        if replacement != rebuilt:
            replacement = complete_indices(rebuilt, replacement)
        out = rebuilt if replacement == rebuilt else visit(replacement)
        active.remove(e)
        done[e] = out
        return out

    return visit(f)
