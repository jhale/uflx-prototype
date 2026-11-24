# Copyright (C) 2026 Jack S. Hale
#
# This file is part of uflx-core-experimental (https://fenics.github.io/uflx/language/)
#
# SPDX-License-Identifier:    MIT
"""Type aliases for UFLx expressions, product arguments and algorithms."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeAlias

from ..nodes import (
    AbstractExpression,
    AbstractExtent,
    AbstractFunctionSpace,
    Base,
    Conjugate,
    Function,
    GatheredIndex,
    Index,
)
from . import Argument

# Values stored on nodes and their index signatures.
Construction: TypeAlias = Base | int | float | str | tuple["Construction", ...]
Extent: TypeAlias = int | AbstractExtent
IndexEntry: TypeAlias = int | Index | GatheredIndex

# One argument can be a single leaf or an ordered product of arguments.
ArgumentSpaces: TypeAlias = tuple[Argument[AbstractFunctionSpace], ...]
FormEntry: TypeAlias = Argument[Function | Conjugate]
BindingEntry: TypeAlias = FormEntry | Index

# Rewriting and lowering.
Transform: TypeAlias = Callable[[AbstractExpression], AbstractExpression]
DerivativeRule: TypeAlias = Callable[
    [AbstractExpression, Transform, Transform], AbstractExpression | None
]
ContractViolation: TypeAlias = tuple[AbstractExpression, str]

# Internal lexical environments and keys for alpha-equivalence.
BoundName: TypeAlias = Function | Index
Names: TypeAlias = tuple[BoundName, ...]
Bindings: TypeAlias = tuple[tuple[BoundName, tuple[int, int]], ...]
Renames: TypeAlias = tuple[tuple[Base, BoundName], ...]
AlphaEnvironment: TypeAlias = tuple[tuple[BoundName, int], ...]
AlphaKey: TypeAlias = tuple[object, ...]
