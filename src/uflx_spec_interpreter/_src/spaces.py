from __future__ import annotations

from typing import overload

from .nodes import AbstractFunctionSpace, ConjugateSpace, DualSpace, ScalarType


def conjugate_space(V: AbstractFunctionSpace) -> AbstractFunctionSpace:
    return (
        V
        if V.scalar_type == ScalarType.real
        else (V.space if isinstance(V, ConjugateSpace) else ConjugateSpace(V))
    )


def dual_space(
    S: AbstractFunctionSpace | tuple[AbstractFunctionSpace, ...],
) -> AbstractFunctionSpace | tuple[AbstractFunctionSpace, ...]:
    return S.space if isinstance(S, DualSpace) else DualSpace(S)


@overload
def conjugated(S: AbstractFunctionSpace) -> AbstractFunctionSpace: ...
@overload
def conjugated(S: tuple[AbstractFunctionSpace, ...]) -> tuple[AbstractFunctionSpace, ...]: ...
def conjugated(
    S: AbstractFunctionSpace | tuple[AbstractFunctionSpace, ...],
) -> AbstractFunctionSpace | tuple[AbstractFunctionSpace, ...]:
    # The conjugate of an argument space, so that conjugating twice gives the space back.
    if isinstance(S, tuple):
        return tuple(conjugated(V) for V in S)
    return S.space if isinstance(S, ConjugateSpace) else conjugate_space(S)
