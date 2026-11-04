from __future__ import annotations

from typing import overload

from .nodes import AbstractFunctionSpace, ConjugateSpace, DualSpace, ScalarType

# Shared implementation helpers; the package public API is declared in __init__.py.
__all__ = ["_conjugated", "conjugate_space", "dual_space"]


def conjugate_space(V: AbstractFunctionSpace) -> AbstractFunctionSpace:
    if V.scalar_type == ScalarType.real:
        return V
    if isinstance(V, ConjugateSpace):
        return V.space
    return ConjugateSpace(V)


def dual_space(
    S: AbstractFunctionSpace | tuple[AbstractFunctionSpace, ...],
) -> AbstractFunctionSpace | tuple[AbstractFunctionSpace, ...]:
    return S.space if isinstance(S, DualSpace) else DualSpace(S)


@overload
def _conjugated(S: AbstractFunctionSpace) -> AbstractFunctionSpace: ...
@overload
def _conjugated(S: tuple[AbstractFunctionSpace, ...]) -> tuple[AbstractFunctionSpace, ...]: ...
def _conjugated(
    S: AbstractFunctionSpace | tuple[AbstractFunctionSpace, ...],
) -> AbstractFunctionSpace | tuple[AbstractFunctionSpace, ...]:
    # The conjugate of an argument space, so that conjugating twice gives the space back.
    if isinstance(S, tuple):
        return tuple(_conjugated(V) for V in S)
    return S.space if isinstance(S, ConjugateSpace) else conjugate_space(S)
