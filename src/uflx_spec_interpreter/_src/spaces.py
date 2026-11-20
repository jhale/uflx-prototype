from __future__ import annotations

from typing import overload

from .nodes import AbstractFunctionSpace, ConjugateSpace, DualSpace, ScalarType


def trivial_conjugation(space: AbstractFunctionSpace) -> bool:
    from .nodes.arguments import argument_members

    return space.scalar_type == ScalarType.real and all(
        trivial_conjugation(member)
        for slot in space.argument_spaces
        for member in argument_members(slot)
    )


def conjugate_space(space: AbstractFunctionSpace) -> AbstractFunctionSpace:
    if trivial_conjugation(space):
        return space
    if isinstance(space, ConjugateSpace):
        return space.space
    return ConjugateSpace(space)


def dual_space(
    space: AbstractFunctionSpace | tuple[AbstractFunctionSpace, ...],
) -> AbstractFunctionSpace | tuple[AbstractFunctionSpace, ...]:
    return space.space if isinstance(space, DualSpace) else DualSpace(space)


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
