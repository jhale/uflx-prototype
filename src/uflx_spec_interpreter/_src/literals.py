from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from .nodes import Complex, Float, Integer, Zero

if TYPE_CHECKING:
    from .nodes.expressions import AbstractExpression
    from .nodes.indices import AbstractExtent, Index
    from .nodes.spaces import AbstractFunctionSpace


# Shared implementation helpers; the package public API is declared in __init__.py.
__all__ = ["_zero", "num"]


def num(value: int | float | complex) -> AbstractExpression:
    # A convenience for the interpreter examples; expression factories never convert numbers.
    if isinstance(value, complex):
        return Complex(Float(value.real), Float(value.imag))
    return Integer(value) if isinstance(value, int) else Float(value)


def _zero(
    value_shape: tuple[int | AbstractExtent, ...],
    free_indices: tuple[tuple[Index, int | AbstractExtent], ...],
    argument_spaces: tuple[AbstractFunctionSpace | tuple[AbstractFunctionSpace, ...], ...],
    scalar_type: Literal["real", "complex"],
) -> Zero:
    return Zero(value_shape, free_indices, argument_spaces, scalar_type)
