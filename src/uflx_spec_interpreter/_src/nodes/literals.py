from __future__ import annotations

import math
from functools import cached_property
from typing import TYPE_CHECKING, final

from .base import AbstractExpression
from .scalar_types import ScalarType

if TYPE_CHECKING:
    from .indices import Index


@final
class Integer(AbstractExpression):
    identifier = "uflx.Integer"
    value: int
    opaque = ("value",)

    def __new__(cls, value, /):
        # An exact int, as rule 1 asks, so not a bool.
        if type(value) is not int:
            raise TypeError(f"Integer takes an int, not {value!r}")
        return super().__new__(cls, value)

    @cached_property
    def value_shape(self):
        return ()

    @cached_property
    def free_indices(self):
        return ()

    @cached_property
    def argument_spaces(self):
        return ()

    @cached_property
    def scalar_type(self):
        return ScalarType.real

    @cached_property
    def domain(self):
        return None


@final
class Float(AbstractExpression):
    identifier = "uflx.Float"
    value: float
    opaque = ("value",)

    def __new__(cls, value, /):
        # An exact finite float, as rule 1 asks.
        if type(value) is not float:
            raise TypeError(f"Float takes a float, not {value!r}")
        if not math.isfinite(value):
            raise ValueError(f"Float takes a finite float, not {value!r}")
        return super().__new__(cls, value)

    @cached_property
    def value_shape(self):
        return ()

    @cached_property
    def free_indices(self):
        return ()

    @cached_property
    def argument_spaces(self):
        return ()

    @cached_property
    def scalar_type(self):
        return ScalarType.real

    @cached_property
    def domain(self):
        return None


@final
class Complex(AbstractExpression):
    identifier = "uflx.Complex"
    real: Integer | Float
    imag: Integer | Float
    opaque = ("real", "imag")

    @cached_property
    def value_shape(self):
        return ()

    @cached_property
    def free_indices(self):
        return ()

    @cached_property
    def argument_spaces(self):
        return ()

    @cached_property
    def scalar_type(self):
        return ScalarType.complex

    @cached_property
    def domain(self):
        return None


@final
class Zero(AbstractExpression):
    identifier = "uflx.Zero"
    # The fields are the static type.
    value_shape: tuple[int, ...]
    free_indices: tuple[tuple[Index, int], ...]
    argument_spaces: tuple
    scalar_type: ScalarType
    opaque = ("value_shape", "free_indices", "argument_spaces", "scalar_type")

    def __new__(cls, value_shape, free_indices, argument_spaces, scalar_type, /):
        if not isinstance(scalar_type, ScalarType):
            raise TypeError("Zero takes a ScalarType")
        return super().__new__(cls, value_shape, free_indices, argument_spaces, scalar_type)

    @cached_property
    def domain(self):
        return None
