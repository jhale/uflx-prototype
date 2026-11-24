# Copyright (C) 2026 Jack S. Hale
#
# This file is part of uflx-core-experimental (https://fenics.github.io/uflx/language/)
#
# SPDX-License-Identifier:    MIT

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, final

from typing_extensions import override

from ..typing import NumericalScalarType
from .caching import cached
from .expressions import AbstractExpression
from .scalar_types import ScalarType, common_scalar_type, numerical_type
from .structure import common_operand_domain

if TYPE_CHECKING:
    from ..typing.nodes import ArgumentSpaces
    from .domains import AbstractDomain
    from .indices import AbstractExtent, Index
    from .topology import AbstractTopology


@final
class Sum(AbstractExpression):
    requires_matching_domains = True
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
        return numerical_type(common_scalar_type(self.first, self.second))

    @property
    def domain(self) -> AbstractDomain | AbstractTopology | None:
        return common_operand_domain(self)

    @override
    def _validate(self) -> None:
        if self.first.value_shape != self.second.value_shape:
            raise ValueError("sum value shapes differ")
        if self.first.argument_spaces != self.second.argument_spaces:
            raise ValueError("sum argument spaces differ")


@final
class Product(AbstractExpression):
    requires_matching_domains = True
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
        first_shape = self.first.value_shape
        return self.second.value_shape if first_shape == () else first_shape

    @property
    @cached
    def free_indices(self) -> tuple[tuple[Index, int | AbstractExtent], ...]:
        return tuple(dict(self.first.free_indices + self.second.free_indices).items())

    @property
    @cached
    def scalar_type(self) -> NumericalScalarType:
        return numerical_type(common_scalar_type(self.first, self.second))

    @property
    def domain(self) -> AbstractDomain | AbstractTopology | None:
        return common_operand_domain(self)

    @override
    def _validate(self) -> None:
        if self.first.value_shape != () and self.second.value_shape != ():
            raise ValueError("product requires at least one scalar operand")


@final
class Division(AbstractExpression):
    requires_matching_domains = True
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
        return numerical_type(common_scalar_type(self.numerator, self.denominator))

    @property
    def domain(self) -> AbstractDomain | AbstractTopology | None:
        return common_operand_domain(self)

    @override
    def _validate(self) -> None:
        if self.denominator.value_shape != ():
            raise ValueError("division requires a scalar denominator")
        if self.denominator.argument_spaces != ():
            raise ValueError("nonlinear operand has argument spaces")


@final
class Power(AbstractExpression):
    requires_matching_domains = True
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
        return numerical_type(common_scalar_type(self.base, self.exponent))

    @property
    def domain(self) -> AbstractDomain | AbstractTopology | None:
        return common_operand_domain(self)

    @override
    def _validate(self) -> None:
        if self.base.value_shape != () or self.exponent.value_shape != ():
            raise ValueError("power requires scalar operands")
        if self.exponent.argument_spaces != ():
            raise ValueError("nonlinear operand has argument spaces")
        if self.base.argument_spaces != ():
            raise ValueError("power base has argument spaces")


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

    @override
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

    @override
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

    @override
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

    @override
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

    @override
    def _validate(self) -> None:
        _scalar_operand(self.operand)


def _scalar_operand(f: AbstractExpression, nonlinear: bool = True) -> None:
    _ = numerical_type(f.scalar_type)
    if f.value_shape != ():
        raise ValueError("operation requires a scalar")
    if nonlinear and f.argument_spaces != ():
        raise ValueError("nonlinear operand has argument spaces")
