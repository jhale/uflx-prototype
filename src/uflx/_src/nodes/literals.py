# Copyright (C) 2026 Jack S. Hale
#
# This file is part of uflx-core-experimental (https://fenics.github.io/uflx/language/)
#
# SPDX-License-Identifier:    MIT

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Literal, cast, final

from ..typing import NumericalScalarType
from .caching import cached
from .expressions import AbstractExpression
from .scalar_types import ScalarType

if TYPE_CHECKING:
    from ..typing.nodes import ArgumentSpaces
    from .domains import AbstractDomain
    from .indices import AbstractExtent, Index
    from .topology import AbstractTopology


@final
class Boolean(AbstractExpression):
    value_fields = ()
    is_terminal = True
    identifier = "uflx.Boolean"
    value: bool

    def _validate(self) -> None:
        if type(self.value) is not bool:
            raise TypeError(f"Boolean takes a bool, not {self.value!r}")

    @property
    @cached
    def value_shape(self) -> tuple[int | AbstractExtent, ...]:
        return ()

    @property
    @cached
    def free_indices(self) -> tuple[tuple[Index, int | AbstractExtent], ...]:
        return ()

    @property
    @cached
    def argument_spaces(self) -> ArgumentSpaces:
        return ()

    @property
    @cached
    def scalar_type(self) -> Literal["boolean"]:
        return ScalarType.boolean

    @property
    @cached
    def domain(self) -> AbstractDomain | AbstractTopology | None:
        return None


@final
class Integer(AbstractExpression):
    value_fields = ()
    is_terminal = True
    identifier = "uflx.Integer"
    value: int

    def _validate(self) -> None:
        if type(self.value) is not int:
            raise TypeError(f"Integer takes an int, not {self.value!r}")

    @property
    @cached
    def value_shape(self) -> tuple[int | AbstractExtent, ...]:
        return ()

    @property
    @cached
    def free_indices(self) -> tuple[tuple[Index, int | AbstractExtent], ...]:
        return ()

    @property
    @cached
    def argument_spaces(self) -> ArgumentSpaces:
        return ()

    @property
    @cached
    def scalar_type(self) -> Literal["real"]:
        return ScalarType.real

    @property
    @cached
    def domain(self) -> AbstractDomain | AbstractTopology | None:
        return None


@final
class Float(AbstractExpression):
    value_fields = ()
    is_terminal = True
    identifier = "uflx.Float"
    value: float

    def _validate(self) -> None:
        if type(self.value) is not float:
            raise TypeError(f"Float takes a float, not {self.value!r}")
        if not math.isfinite(self.value):
            raise ValueError(f"Float takes a finite float, not {self.value!r}")

    @property
    @cached
    def value_shape(self) -> tuple[int | AbstractExtent, ...]:
        return ()

    @property
    @cached
    def free_indices(self) -> tuple[tuple[Index, int | AbstractExtent], ...]:
        return ()

    @property
    @cached
    def argument_spaces(self) -> ArgumentSpaces:
        return ()

    @property
    @cached
    def scalar_type(self) -> Literal["real"]:
        return ScalarType.real

    @property
    @cached
    def domain(self) -> AbstractDomain | AbstractTopology | None:
        return None


@final
class Complex(AbstractExpression):
    value_fields = ()
    is_terminal = True
    identifier = "uflx.Complex"
    real: Integer | Float
    imag: Integer | Float

    @property
    @cached
    def value_shape(self) -> tuple[int | AbstractExtent, ...]:
        return ()

    @property
    @cached
    def free_indices(self) -> tuple[tuple[Index, int | AbstractExtent], ...]:
        return ()

    @property
    @cached
    def argument_spaces(self) -> ArgumentSpaces:
        return ()

    @property
    @cached
    def scalar_type(self) -> Literal["complex"]:
        return ScalarType.complex

    @property
    @cached
    def domain(self) -> AbstractDomain | AbstractTopology | None:
        return None


@final
class Zero(AbstractExpression):
    value_fields = ()
    is_terminal = True
    identifier = "uflx.Zero"
    # The fields are the static type.

    _value_shape: tuple[int | AbstractExtent, ...]
    _free_indices: tuple[tuple[Index, int | AbstractExtent], ...]
    _argument_spaces: ArgumentSpaces
    _scalar_type: NumericalScalarType

    @property
    def value_shape(self) -> tuple[int | AbstractExtent, ...]:
        return self._value_shape

    @property
    def free_indices(self) -> tuple[tuple[Index, int | AbstractExtent], ...]:
        return self._free_indices

    @property
    def argument_spaces(self) -> ArgumentSpaces:
        return self._argument_spaces

    @property
    def scalar_type(self) -> NumericalScalarType:
        return self._scalar_type

    @property
    @cached
    def domain(self) -> AbstractDomain | AbstractTopology | None:
        return None

    def _validate(self) -> None:
        if cast("object", self.scalar_type) == ScalarType.boolean:
            raise ValueError("Boolean zero is undefined")
        if type(self.scalar_type) is not str or self.scalar_type not in (
            ScalarType.real,
            ScalarType.complex,
        ):
            raise TypeError("Zero takes a real or complex scalar type")
        from .validation import extent, product_scalar_type, require

        for size in self.value_shape:
            extent(size)
        seen: set[Index] = set()
        for pair in self.free_indices:
            require(len(pair) == 2, "a free index is an index/extent pair")
            i, size = pair
            extent(size)
            require(i not in seen, "free index is repeated")
            seen.add(i)
        for space in self.argument_spaces:
            product_scalar_type(space)
