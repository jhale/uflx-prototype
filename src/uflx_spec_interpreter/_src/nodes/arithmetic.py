from __future__ import annotations

from typing import TYPE_CHECKING, Literal, final

from .caching import _cached
from .expressions import AbstractExpression
from .scalar_types import ScalarType
from .validation import _require

if TYPE_CHECKING:
    from .domains import AbstractDomain
    from .indices import AbstractExtent, Index
    from .spaces import AbstractFunctionSpace
    from .topology import AbstractTopology


@final
class Sum(AbstractExpression):
    matching_domains = True
    value_fields = ("first", "second")
    identifier = "uflx.Sum"
    first: AbstractExpression
    second: AbstractExpression

    @property
    @_cached
    def value_shape(self) -> tuple[int | AbstractExtent, ...]:
        return self.first.value_shape

    @property
    @_cached
    def free_indices(self) -> tuple[tuple[Index, int | AbstractExtent], ...]:
        return tuple(dict(self.first.free_indices + self.second.free_indices).items())

    @property
    @_cached
    def argument_spaces(
        self,
    ) -> tuple[AbstractFunctionSpace | tuple[AbstractFunctionSpace, ...], ...]:
        return self.first.argument_spaces

    @property
    @_cached
    def scalar_type(self) -> Literal["real", "complex"]:
        return self.joined_scalar_type(self.first, self.second)

    @property
    @_cached
    def domain(self) -> AbstractDomain | AbstractTopology | None:
        return self.second.domain or self.first.domain

    def _validate(self) -> None:
        _require(self.first.value_shape == self.second.value_shape, "sum value shapes differ")
        _require(
            self.first.argument_spaces == self.second.argument_spaces, "sum argument spaces differ"
        )


@final
class Product(AbstractExpression):
    matching_domains = True
    value_fields = ("first", "second")
    identifier = "uflx.Product"
    first: AbstractExpression
    second: AbstractExpression

    @property
    @_cached
    def argument_spaces(
        self,
    ) -> tuple[AbstractFunctionSpace | tuple[AbstractFunctionSpace, ...], ...]:
        return self.first.argument_spaces + self.second.argument_spaces

    @property
    @_cached
    def value_shape(self) -> tuple[int | AbstractExtent, ...]:
        return self.first.value_shape or self.second.value_shape

    @property
    @_cached
    def free_indices(self) -> tuple[tuple[Index, int | AbstractExtent], ...]:
        return tuple(dict(self.first.free_indices + self.second.free_indices).items())

    @property
    @_cached
    def scalar_type(self) -> Literal["real", "complex"]:
        return self.joined_scalar_type(self.first, self.second)

    @property
    @_cached
    def domain(self) -> AbstractDomain | AbstractTopology | None:
        return self.second.domain or self.first.domain

    def _validate(self) -> None:
        _require(
            not self.first.value_shape or not self.second.value_shape,
            "product requires at least one scalar operand",
        )


@final
class Division(AbstractExpression):
    matching_domains = True
    value_fields = ("first", "second")
    identifier = "uflx.Division"
    first: AbstractExpression
    second: AbstractExpression

    @property
    @_cached
    def value_shape(self) -> tuple[int | AbstractExtent, ...]:
        return self.first.value_shape

    @property
    @_cached
    def free_indices(self) -> tuple[tuple[Index, int | AbstractExtent], ...]:
        return tuple(dict(self.first.free_indices + self.second.free_indices).items())

    @property
    @_cached
    def argument_spaces(
        self,
    ) -> tuple[AbstractFunctionSpace | tuple[AbstractFunctionSpace, ...], ...]:
        return self.first.argument_spaces

    @property
    @_cached
    def scalar_type(self) -> Literal["real", "complex"]:
        return self.joined_scalar_type(self.first, self.second)

    @property
    @_cached
    def domain(self) -> AbstractDomain | AbstractTopology | None:
        return self.second.domain or self.first.domain

    def _validate(self) -> None:
        _require(not self.second.value_shape, "division requires a scalar denominator")
        _require(not self.second.argument_spaces, "nonlinear operand has argument spaces")


@final
class Power(AbstractExpression):
    matching_domains = True
    value_fields = ("first", "second")
    identifier = "uflx.Power"
    first: AbstractExpression
    second: AbstractExpression

    @property
    @_cached
    def value_shape(self) -> tuple[int | AbstractExtent, ...]:
        return ()

    @property
    @_cached
    def free_indices(self) -> tuple[tuple[Index, int | AbstractExtent], ...]:
        return tuple(dict(self.first.free_indices + self.second.free_indices).items())

    @property
    @_cached
    def argument_spaces(
        self,
    ) -> tuple[AbstractFunctionSpace | tuple[AbstractFunctionSpace, ...], ...]:
        return ()

    @property
    @_cached
    def scalar_type(self) -> Literal["real", "complex"]:
        return self.joined_scalar_type(self.first, self.second)

    @property
    @_cached
    def domain(self) -> AbstractDomain | AbstractTopology | None:
        return self.second.domain or self.first.domain

    def _validate(self) -> None:
        _require(
            not self.first.value_shape and not self.second.value_shape,
            "power requires scalar operands",
        )
        _require(not self.second.argument_spaces, "nonlinear operand has argument spaces")
        _require(not self.first.argument_spaces, "power base has argument spaces")


@final
class Conjugate(AbstractExpression):
    value_fields = ("operand",)
    identifier = "uflx.Conjugate"
    operand: AbstractExpression

    @property
    @_cached
    def value_shape(self) -> tuple[int | AbstractExtent, ...]:
        return self.operand.value_shape

    @property
    @_cached
    def free_indices(self) -> tuple[tuple[Index, int | AbstractExtent], ...]:
        return self.operand.free_indices

    @property
    @_cached
    def argument_spaces(
        self,
    ) -> tuple[AbstractFunctionSpace | tuple[AbstractFunctionSpace, ...], ...]:
        # The free functions import the nodes, so the import is local.
        from ..spaces import _conjugated

        return tuple(_conjugated(V) for V in self.operand.argument_spaces)

    @property
    @_cached
    def scalar_type(self) -> Literal["real", "complex"]:
        return self.joined_scalar_type(self.operand)

    @property
    @_cached
    def domain(self) -> AbstractDomain | AbstractTopology | None:
        return self.operand.domain


@final
class RealPart(AbstractExpression):
    value_fields = ("operand",)
    identifier = "uflx.RealPart"
    operand: AbstractExpression

    @property
    @_cached
    def value_shape(self) -> tuple[int | AbstractExtent, ...]:
        return ()

    @property
    @_cached
    def free_indices(self) -> tuple[tuple[Index, int | AbstractExtent], ...]:
        return self.operand.free_indices

    @property
    @_cached
    def argument_spaces(
        self,
    ) -> tuple[AbstractFunctionSpace | tuple[AbstractFunctionSpace, ...], ...]:
        return ()

    @property
    @_cached
    def scalar_type(self) -> Literal["real", "complex"]:
        return ScalarType.real

    @property
    @_cached
    def domain(self) -> AbstractDomain | AbstractTopology | None:
        return self.operand.domain

    def _validate(self) -> None:
        from .validation import _scalar_operand

        _scalar_operand(self.operand)


@final
class ImaginaryPart(AbstractExpression):
    value_fields = ("operand",)
    identifier = "uflx.ImaginaryPart"
    operand: AbstractExpression

    @property
    @_cached
    def value_shape(self) -> tuple[int | AbstractExtent, ...]:
        return ()

    @property
    @_cached
    def free_indices(self) -> tuple[tuple[Index, int | AbstractExtent], ...]:
        return self.operand.free_indices

    @property
    @_cached
    def argument_spaces(
        self,
    ) -> tuple[AbstractFunctionSpace | tuple[AbstractFunctionSpace, ...], ...]:
        return ()

    @property
    @_cached
    def scalar_type(self) -> Literal["real", "complex"]:
        return ScalarType.real

    @property
    @_cached
    def domain(self) -> AbstractDomain | AbstractTopology | None:
        return self.operand.domain

    def _validate(self) -> None:
        from .validation import _scalar_operand

        _scalar_operand(self.operand)


@final
class Absolute(AbstractExpression):
    value_fields = ("operand",)
    identifier = "uflx.Absolute"
    operand: AbstractExpression

    @property
    @_cached
    def value_shape(self) -> tuple[int | AbstractExtent, ...]:
        return ()

    @property
    @_cached
    def free_indices(self) -> tuple[tuple[Index, int | AbstractExtent], ...]:
        return self.operand.free_indices

    @property
    @_cached
    def argument_spaces(
        self,
    ) -> tuple[AbstractFunctionSpace | tuple[AbstractFunctionSpace, ...], ...]:
        return ()

    @property
    @_cached
    def scalar_type(self) -> Literal["real", "complex"]:
        return ScalarType.real

    @property
    @_cached
    def domain(self) -> AbstractDomain | AbstractTopology | None:
        return self.operand.domain

    def _validate(self) -> None:
        from .validation import _scalar_operand

        _scalar_operand(self.operand)


@final
class SquareRoot(AbstractExpression):
    value_fields = ("operand",)
    identifier = "uflx.SquareRoot"
    operand: AbstractExpression

    @property
    @_cached
    def value_shape(self) -> tuple[int | AbstractExtent, ...]:
        return ()

    @property
    @_cached
    def free_indices(self) -> tuple[tuple[Index, int | AbstractExtent], ...]:
        return self.operand.free_indices

    @property
    @_cached
    def argument_spaces(
        self,
    ) -> tuple[AbstractFunctionSpace | tuple[AbstractFunctionSpace, ...], ...]:
        return ()

    @property
    @_cached
    def scalar_type(self) -> Literal["real", "complex"]:
        return self.joined_scalar_type(self.operand)

    @property
    @_cached
    def domain(self) -> AbstractDomain | AbstractTopology | None:
        return self.operand.domain

    def _validate(self) -> None:
        from .validation import _scalar_operand

        _scalar_operand(self.operand)


@final
class Logarithm(AbstractExpression):
    value_fields = ("operand",)
    identifier = "uflx.Logarithm"
    operand: AbstractExpression

    @property
    @_cached
    def value_shape(self) -> tuple[int | AbstractExtent, ...]:
        return ()

    @property
    @_cached
    def free_indices(self) -> tuple[tuple[Index, int | AbstractExtent], ...]:
        return self.operand.free_indices

    @property
    @_cached
    def argument_spaces(
        self,
    ) -> tuple[AbstractFunctionSpace | tuple[AbstractFunctionSpace, ...], ...]:
        return ()

    @property
    @_cached
    def scalar_type(self) -> Literal["real", "complex"]:
        return self.joined_scalar_type(self.operand)

    @property
    @_cached
    def domain(self) -> AbstractDomain | AbstractTopology | None:
        return self.operand.domain

    def _validate(self) -> None:
        from .validation import _scalar_operand

        _scalar_operand(self.operand)
