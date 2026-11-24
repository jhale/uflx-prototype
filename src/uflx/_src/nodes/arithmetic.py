# Copyright (C) 2026 Jack S. Hale
#
# This file is part of uflx-core-experimental (https://fenics.github.io/uflx/language/)
#
# SPDX-License-Identifier:    MIT

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, final

from .arguments import ArgumentSpaces
from .caching import cached
from .expressions import AbstractExpression
from .scalar_types import NumericalScalarType, ScalarType, numerical_type
from .validation import require

if TYPE_CHECKING:
    from .domains import AbstractDomain
    from .indices import AbstractExtent, Index
    from .topology import AbstractTopology


@final
class Sum(AbstractExpression):
    matching_domains = True
    value_fields = ("first", "second")
    identifier = "uflx.Sum"
    first: AbstractExpression
    second: AbstractExpression

    @property
    @cached
    def value_shape(self) -> tuple[int | AbstractExtent, ...]:
        return self.first.value_shape

    @property
    @cached
    def free_indices(self) -> tuple[tuple[Index, int | AbstractExtent], ...]:
        return tuple(dict(self.first.free_indices + self.second.free_indices).items())

    @property
    @cached
    def argument_spaces(self) -> ArgumentSpaces:
        return self.first.argument_spaces

    @property
    @cached
    def scalar_type(self) -> NumericalScalarType:
        return numerical_type(self.joined_scalar_type(self.first, self.second))

    @property
    @cached
    def domain(self) -> AbstractDomain | AbstractTopology | None:
        return self.second.domain or self.first.domain

    def _validate(self) -> None:
        require(self.first.value_shape == self.second.value_shape, "sum value shapes differ")
        require(
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
    @cached
    def argument_spaces(self) -> ArgumentSpaces:
        return self.first.argument_spaces + self.second.argument_spaces

    @property
    @cached
    def value_shape(self) -> tuple[int | AbstractExtent, ...]:
        return self.first.value_shape or self.second.value_shape

    @property
    @cached
    def free_indices(self) -> tuple[tuple[Index, int | AbstractExtent], ...]:
        return tuple(dict(self.first.free_indices + self.second.free_indices).items())

    @property
    @cached
    def scalar_type(self) -> NumericalScalarType:
        return numerical_type(self.joined_scalar_type(self.first, self.second))

    @property
    @cached
    def domain(self) -> AbstractDomain | AbstractTopology | None:
        return self.second.domain or self.first.domain

    def _validate(self) -> None:
        require(
            not self.first.value_shape or not self.second.value_shape,
            "product requires at least one scalar operand",
        )


@final
class Division(AbstractExpression):
    matching_domains = True
    value_fields = ("numerator", "denominator")
    identifier = "uflx.Division"
    numerator: AbstractExpression
    denominator: AbstractExpression

    @property
    @cached
    def value_shape(self) -> tuple[int | AbstractExtent, ...]:
        return self.numerator.value_shape

    @property
    @cached
    def free_indices(self) -> tuple[tuple[Index, int | AbstractExtent], ...]:
        return tuple(dict(self.numerator.free_indices + self.denominator.free_indices).items())

    @property
    @cached
    def argument_spaces(self) -> ArgumentSpaces:
        return self.numerator.argument_spaces

    @property
    @cached
    def scalar_type(self) -> NumericalScalarType:
        return numerical_type(self.joined_scalar_type(self.numerator, self.denominator))

    @property
    @cached
    def domain(self) -> AbstractDomain | AbstractTopology | None:
        return self.denominator.domain or self.numerator.domain

    def _validate(self) -> None:
        require(not self.denominator.value_shape, "division requires a scalar denominator")
        require(not self.denominator.argument_spaces, "nonlinear operand has argument spaces")


@final
class Power(AbstractExpression):
    matching_domains = True
    value_fields = ("base", "exponent")
    identifier = "uflx.Power"
    base: AbstractExpression
    exponent: AbstractExpression

    @property
    @cached
    def value_shape(self) -> tuple[int | AbstractExtent, ...]:
        return ()

    @property
    @cached
    def free_indices(self) -> tuple[tuple[Index, int | AbstractExtent], ...]:
        return tuple(dict(self.base.free_indices + self.exponent.free_indices).items())

    @property
    @cached
    def argument_spaces(self) -> ArgumentSpaces:
        return ()

    @property
    @cached
    def scalar_type(self) -> NumericalScalarType:
        return numerical_type(self.joined_scalar_type(self.base, self.exponent))

    @property
    @cached
    def domain(self) -> AbstractDomain | AbstractTopology | None:
        return self.exponent.domain or self.base.domain

    def _validate(self) -> None:
        require(
            not self.base.value_shape and not self.exponent.value_shape,
            "power requires scalar operands",
        )
        require(not self.exponent.argument_spaces, "nonlinear operand has argument spaces")
        require(not self.base.argument_spaces, "power base has argument spaces")


@final
class Conjugate(AbstractExpression):
    value_fields = ("operand",)
    identifier = "uflx.Conjugate"
    operand: AbstractExpression

    @property
    @cached
    def value_shape(self) -> tuple[int | AbstractExtent, ...]:
        return self.operand.value_shape

    @property
    @cached
    def free_indices(self) -> tuple[tuple[Index, int | AbstractExtent], ...]:
        return self.operand.free_indices

    @property
    @cached
    def argument_spaces(self) -> ArgumentSpaces:
        # The free functions import the nodes, so the import is local.
        from ..spaces import conjugated

        return tuple(conjugated(V) for V in self.operand.argument_spaces)

    @property
    @cached
    def scalar_type(self) -> NumericalScalarType:
        return numerical_type(self.operand.scalar_type)

    @property
    @cached
    def domain(self) -> AbstractDomain | AbstractTopology | None:
        return self.operand.domain


@final
class RealPart(AbstractExpression):
    value_fields = ("operand",)
    identifier = "uflx.RealPart"
    operand: AbstractExpression

    @property
    @cached
    def value_shape(self) -> tuple[int | AbstractExtent, ...]:
        return ()

    @property
    @cached
    def free_indices(self) -> tuple[tuple[Index, int | AbstractExtent], ...]:
        return self.operand.free_indices

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
        return self.operand.domain

    def _validate(self) -> None:
        _scalar_operand(self.operand)


@final
class ImaginaryPart(AbstractExpression):
    value_fields = ("operand",)
    identifier = "uflx.ImaginaryPart"
    operand: AbstractExpression

    @property
    @cached
    def value_shape(self) -> tuple[int | AbstractExtent, ...]:
        return ()

    @property
    @cached
    def free_indices(self) -> tuple[tuple[Index, int | AbstractExtent], ...]:
        return self.operand.free_indices

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
        return self.operand.domain

    def _validate(self) -> None:
        _scalar_operand(self.operand)


@final
class Absolute(AbstractExpression):
    value_fields = ("operand",)
    identifier = "uflx.Absolute"
    operand: AbstractExpression

    @property
    @cached
    def value_shape(self) -> tuple[int | AbstractExtent, ...]:
        return ()

    @property
    @cached
    def free_indices(self) -> tuple[tuple[Index, int | AbstractExtent], ...]:
        return self.operand.free_indices

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
        return self.operand.domain

    def _validate(self) -> None:
        _scalar_operand(self.operand)


@final
class SquareRoot(AbstractExpression):
    value_fields = ("operand",)
    identifier = "uflx.SquareRoot"
    operand: AbstractExpression

    @property
    @cached
    def value_shape(self) -> tuple[int | AbstractExtent, ...]:
        return ()

    @property
    @cached
    def free_indices(self) -> tuple[tuple[Index, int | AbstractExtent], ...]:
        return self.operand.free_indices

    @property
    @cached
    def argument_spaces(self) -> ArgumentSpaces:
        return ()

    @property
    @cached
    def scalar_type(self) -> NumericalScalarType:
        return numerical_type(self.operand.scalar_type)

    @property
    @cached
    def domain(self) -> AbstractDomain | AbstractTopology | None:
        return self.operand.domain

    def _validate(self) -> None:
        _scalar_operand(self.operand)


@final
class Logarithm(AbstractExpression):
    value_fields = ("operand",)
    identifier = "uflx.Logarithm"
    operand: AbstractExpression

    @property
    @cached
    def value_shape(self) -> tuple[int | AbstractExtent, ...]:
        return ()

    @property
    @cached
    def free_indices(self) -> tuple[tuple[Index, int | AbstractExtent], ...]:
        return self.operand.free_indices

    @property
    @cached
    def argument_spaces(self) -> ArgumentSpaces:
        return ()

    @property
    @cached
    def scalar_type(self) -> NumericalScalarType:
        return numerical_type(self.operand.scalar_type)

    @property
    @cached
    def domain(self) -> AbstractDomain | AbstractTopology | None:
        return self.operand.domain

    def _validate(self) -> None:
        _scalar_operand(self.operand)


def _scalar_operand(f: AbstractExpression, nonlinear: bool = True) -> None:
    numerical_type(f.scalar_type)
    require(f.value_shape == (), "operation requires a scalar")
    if nonlinear:
        require(not f.argument_spaces, "nonlinear operand has argument spaces")
