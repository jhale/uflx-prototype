# Copyright (C) 2026 Jack S. Hale
#
# This file is part of uflx-core-experimental (https://fenics.github.io/uflx/language/)
#
# SPDX-License-Identifier:    MIT
"""Named real arrays supplied at evaluation time."""

from typing import Literal, final

from typing_extensions import override

from uflx.nodes import AbstractExpression, AbstractExtent, ScalarType
from uflx.typing import Extent

__all__ = ["RuntimeArray", "RuntimeExtent"]


@final
class RuntimeExtent(AbstractExtent):
    """A named axis length that can be shared by several array inputs."""

    identifier = "uflx_extensions.RuntimeExtent"
    label: str


@final
class RuntimeArray(AbstractExpression):
    """A domain-independent real array with fixed or runtime axis lengths."""

    identifier = "uflx_extensions.RuntimeArray"
    is_terminal = True
    label: str
    _value_shape: tuple[Extent, ...]

    @property
    def value_shape(self) -> tuple[Extent, ...]:
        return self._value_shape

    @property
    def scalar_type(self) -> Literal["real"]:
        return ScalarType.real

    @property
    def domain(self) -> None:
        return None

    @property
    def free_indices(self) -> tuple[()]:
        return ()

    @property
    def argument_spaces(self) -> tuple[()]:
        return ()

    @override
    def _validate(self) -> None:
        for n in self.value_shape:
            if isinstance(n, AbstractExtent):
                continue
            if type(n) is not int:
                raise TypeError("expected an integer or an abstract extent")
            if n < 0:
                raise ValueError("array extents must be nonnegative")
