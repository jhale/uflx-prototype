from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING, final

from .base import AbstractExpression
from .scalar_types import ScalarType

if TYPE_CHECKING:
    pass


@final
class Sum(AbstractExpression):
    identifier = "uflx.Sum"
    first: AbstractExpression
    second: AbstractExpression

    @cached_property
    def value_shape(self):
        return self.first.value_shape

    @cached_property
    def free_indices(self):
        return tuple(dict(self.first.free_indices + self.second.free_indices).items())

    @cached_property
    def argument_spaces(self):
        return self.first.argument_spaces

    @cached_property
    def scalar_type(self):
        return self.joined_scalar_type(self.first, self.second)

    @cached_property
    def support(self):
        return self.second.support or self.first.support


@final
class Product(AbstractExpression):
    identifier = "uflx.Product"
    first: AbstractExpression
    second: AbstractExpression

    @cached_property
    def argument_spaces(self):
        return self.first.argument_spaces + self.second.argument_spaces

    @cached_property
    def value_shape(self):
        return ()

    @cached_property
    def free_indices(self):
        return tuple(dict(self.first.free_indices + self.second.free_indices).items())

    @cached_property
    def scalar_type(self):
        return self.joined_scalar_type(self.first, self.second)

    @cached_property
    def support(self):
        return self.second.support or self.first.support


@final
class Division(AbstractExpression):
    identifier = "uflx.Division"
    first: AbstractExpression
    second: AbstractExpression

    @cached_property
    def value_shape(self):
        return ()

    @cached_property
    def free_indices(self):
        return tuple(dict(self.first.free_indices + self.second.free_indices).items())

    @cached_property
    def argument_spaces(self):
        return self.first.argument_spaces

    @cached_property
    def scalar_type(self):
        return self.joined_scalar_type(self.first, self.second)

    @cached_property
    def support(self):
        return self.second.support or self.first.support


@final
class Power(AbstractExpression):
    identifier = "uflx.Power"
    first: AbstractExpression
    second: AbstractExpression

    @cached_property
    def value_shape(self):
        return ()

    @cached_property
    def free_indices(self):
        return tuple(dict(self.first.free_indices + self.second.free_indices).items())

    @cached_property
    def argument_spaces(self):
        return ()

    @cached_property
    def scalar_type(self):
        return self.joined_scalar_type(self.first, self.second)

    @cached_property
    def support(self):
        return self.second.support or self.first.support


@final
class Conjugate(AbstractExpression):
    identifier = "uflx.Conjugate"
    operand: AbstractExpression

    @cached_property
    def value_shape(self):
        return self.operand.value_shape

    @cached_property
    def free_indices(self):
        return self.operand.free_indices

    @cached_property
    def argument_spaces(self):
        # The free functions import the nodes, so the import is local.
        from ..spaces import conjugated

        return conjugated(self.operand.argument_spaces)

    @cached_property
    def scalar_type(self):
        return self.joined_scalar_type(self.operand)

    @cached_property
    def support(self):
        return self.operand.support


@final
class RealPart(AbstractExpression):
    identifier = "uflx.RealPart"
    operand: AbstractExpression

    @cached_property
    def value_shape(self):
        return ()

    @cached_property
    def free_indices(self):
        return self.operand.free_indices

    @cached_property
    def argument_spaces(self):
        return ()

    @cached_property
    def scalar_type(self):
        return ScalarType.real

    @cached_property
    def support(self):
        return self.operand.support


@final
class ImaginaryPart(AbstractExpression):
    identifier = "uflx.ImaginaryPart"
    operand: AbstractExpression

    @cached_property
    def value_shape(self):
        return ()

    @cached_property
    def free_indices(self):
        return self.operand.free_indices

    @cached_property
    def argument_spaces(self):
        return ()

    @cached_property
    def scalar_type(self):
        return ScalarType.real

    @cached_property
    def support(self):
        return self.operand.support


@final
class Absolute(AbstractExpression):
    identifier = "uflx.Absolute"
    operand: AbstractExpression

    @cached_property
    def value_shape(self):
        return ()

    @cached_property
    def free_indices(self):
        return self.operand.free_indices

    @cached_property
    def argument_spaces(self):
        return ()

    @cached_property
    def scalar_type(self):
        return ScalarType.real

    @cached_property
    def support(self):
        return self.operand.support


@final
class SquareRoot(AbstractExpression):
    identifier = "uflx.SquareRoot"
    operand: AbstractExpression

    @cached_property
    def value_shape(self):
        return ()

    @cached_property
    def free_indices(self):
        return self.operand.free_indices

    @cached_property
    def argument_spaces(self):
        return ()

    @cached_property
    def scalar_type(self):
        return self.joined_scalar_type(self.operand)

    @cached_property
    def support(self):
        return self.operand.support


@final
class Logarithm(AbstractExpression):
    identifier = "uflx.Logarithm"
    operand: AbstractExpression

    @cached_property
    def value_shape(self):
        return ()

    @cached_property
    def free_indices(self):
        return self.operand.free_indices

    @cached_property
    def argument_spaces(self):
        return ()

    @cached_property
    def scalar_type(self):
        return self.joined_scalar_type(self.operand)

    @cached_property
    def support(self):
        return self.operand.support
