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
    value_fields = ()
    is_terminal = True
    identifier = "uflx.Integer"
    value: int

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
    value_fields = ()
    is_terminal = True
    identifier = "uflx.Float"
    value: float

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
    value_fields = ()
    is_terminal = True
    identifier = "uflx.Complex"
    real: Integer | Float
    imag: Integer | Float

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

    def _validate(self):
        from ..validation import typed

        typed(self.real, (Integer, Float), "complex parts must be real literals")
        typed(self.imag, (Integer, Float), "complex parts must be real literals")


@final
class Zero(AbstractExpression):
    value_fields = ()
    is_terminal = True
    identifier = "uflx.Zero"
    # The fields are the static type.
    value_shape: tuple[int, ...]
    free_indices: tuple[tuple[Index, int], ...]
    argument_spaces: tuple
    scalar_type: str

    def __new__(cls, value_shape, free_indices, argument_spaces, scalar_type, /):
        if type(scalar_type) is not str or scalar_type not in (ScalarType.real, ScalarType.complex):
            raise TypeError("Zero takes a real or complex scalar type")
        return super().__new__(cls, value_shape, free_indices, argument_spaces, scalar_type)

    @cached_property
    def domain(self):
        return None

    def _validate(self):
        from ..validation import extent, require, tuple_of, typed
        from .indices import AbstractExtent, Index
        from .spaces import AbstractFunctionSpace

        tuple_of(self.value_shape, (int, AbstractExtent))
        for size in self.value_shape:
            extent(size)
        tuple_of(self.free_indices, tuple)
        seen = set()
        for pair in self.free_indices:
            require(len(pair) == 2, "a free index is an index/extent pair")
            i, size = pair
            typed(i, Index, "free index must be an Index")
            extent(size)
            require(i not in seen, "free index is repeated")
            seen.add(i)
        tuple_of(self.argument_spaces, (AbstractFunctionSpace, tuple))
        for V in self.argument_spaces:
            if type(V) is tuple:
                tuple_of(V, AbstractFunctionSpace)
                require(len(V) >= 2, "product argument needs at least two spaces")
