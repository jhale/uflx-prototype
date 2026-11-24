# Copyright (C) 2026 Jack S. Hale
#
# This file is part of uflx-core-experimental (https://fenics.github.io/uflx/language/)
#
# SPDX-License-Identifier:    MIT

from __future__ import annotations

from typing import overload

from .nodes import AbstractFunctionSpace, ConjugateSpace, DualSpace, ScalarType
from .nodes.arguments import argument_leaves, map_argument
from .typing import Argument


def trivial_conjugation(space: AbstractFunctionSpace) -> bool:
    return space.scalar_type == ScalarType.real and all(
        trivial_conjugation(member)
        for slot in space.argument_spaces
        for member in argument_leaves(slot)
    )


def conjugate_space(space: AbstractFunctionSpace) -> AbstractFunctionSpace:
    if isinstance(space, ConjugateSpace):
        return space.space
    if trivial_conjugation(space):
        return space
    return ConjugateSpace(space)


def dual_space(
    space: Argument[AbstractFunctionSpace],
) -> Argument[AbstractFunctionSpace]:
    return space.space if isinstance(space, DualSpace) else DualSpace(space)


@overload
def conjugated(S: AbstractFunctionSpace) -> AbstractFunctionSpace: ...
@overload
def conjugated(
    S: tuple[Argument[AbstractFunctionSpace], ...],
) -> tuple[Argument[AbstractFunctionSpace], ...]: ...
def conjugated(
    S: Argument[AbstractFunctionSpace],
) -> Argument[AbstractFunctionSpace]:
    # The conjugate of an argument space, so that conjugating twice gives the space back.
    return map_argument(conjugate_space, S)
