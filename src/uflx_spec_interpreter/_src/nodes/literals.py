from __future__ import annotations

import math
from typing import TYPE_CHECKING, Literal, Self, final

from ..interning import cached
from .base import AbstractExpression
from .scalar_types import ScalarType

if TYPE_CHECKING:
    from ..nodes.domains import AbstractDomain
    from ..nodes.indices import AbstractExtent, Index
    from ..nodes.spaces import AbstractFunctionSpace
    from ..nodes.topology import AbstractTopology


@final
class Integer(AbstractExpression):
    value_fields = ()
    is_terminal = True
    identifier = "uflx.Integer"
    value: int

    def __new__(cls, value: int, /) -> Self:
        # An exact int, as rule 1 asks, so not a bool.
        if type(value) is not int:
            raise TypeError(f"Integer takes an int, not {value!r}")
        return super().__new__(cls, value)

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
    def argument_spaces(
        self,
    ) -> tuple[AbstractFunctionSpace | tuple[AbstractFunctionSpace, ...], ...]:
        return ()

    @property
    @cached
    def scalar_type(self) -> Literal["real", "complex"]:
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

    def __new__(cls, value: float, /) -> Self:
        # An exact finite float, as rule 1 asks.
        if type(value) is not float:
            raise TypeError(f"Float takes a float, not {value!r}")
        if not math.isfinite(value):
            raise ValueError(f"Float takes a finite float, not {value!r}")
        return super().__new__(cls, value)

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
    def argument_spaces(
        self,
    ) -> tuple[AbstractFunctionSpace | tuple[AbstractFunctionSpace, ...], ...]:
        return ()

    @property
    @cached
    def scalar_type(self) -> Literal["real", "complex"]:
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
    def argument_spaces(
        self,
    ) -> tuple[AbstractFunctionSpace | tuple[AbstractFunctionSpace, ...], ...]:
        return ()

    @property
    @cached
    def scalar_type(self) -> Literal["real", "complex"]:
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
    value_shape: tuple[int | AbstractExtent, ...]  # pyright: ignore[reportIncompatibleMethodOverride]
    free_indices: tuple[tuple[Index, int | AbstractExtent], ...]  # pyright: ignore[reportIncompatibleMethodOverride]
    argument_spaces: tuple[AbstractFunctionSpace | tuple[AbstractFunctionSpace, ...], ...]  # pyright: ignore[reportIncompatibleMethodOverride]
    scalar_type: Literal["real", "complex"]  # pyright: ignore[reportIncompatibleMethodOverride]

    def __new__(
        cls,
        value_shape: tuple[int | AbstractExtent, ...],
        free_indices: tuple[tuple[Index, int | AbstractExtent], ...],
        argument_spaces: tuple[AbstractFunctionSpace | tuple[AbstractFunctionSpace, ...], ...],
        scalar_type: Literal["real", "complex"],
        /,
    ) -> Self:
        if type(scalar_type) is not str or scalar_type not in (ScalarType.real, ScalarType.complex):
            raise TypeError("Zero takes a real or complex scalar type")
        return super().__new__(cls, value_shape, free_indices, argument_spaces, scalar_type)

    @property
    @cached
    def domain(self) -> AbstractDomain | AbstractTopology | None:
        return None

    def _validate(self) -> None:
        from ..validation import extent, require

        for size in self.value_shape:
            extent(size)
        seen: set[Index] = set()
        for pair in self.free_indices:
            require(len(pair) == 2, "a free index is an index/extent pair")
            i, size = pair
            extent(size)
            require(i not in seen, "free index is repeated")
            seen.add(i)
        for V in self.argument_spaces:
            if isinstance(V, tuple):
                require(len(V) >= 2, "product argument needs at least two spaces")
