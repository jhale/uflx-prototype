from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING, final

from .base import AbstractExpression

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
    def arity(self):
        return self.first.arity

    @cached_property
    def support(self):
        return self.second.support or self.first.support


@final
class Product(AbstractExpression):
    identifier = "uflx.Product"
    first: AbstractExpression
    second: AbstractExpression

    @cached_property
    def arity(self):
        return self.first.arity + self.second.arity

    @cached_property
    def value_shape(self):
        return ()

    @cached_property
    def free_indices(self):
        return tuple(dict(self.first.free_indices + self.second.free_indices).items())

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
    def arity(self):
        return self.first.arity

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
    def arity(self):
        return self.first.arity

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
    def arity(self):
        return self.operand.arity

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
    def arity(self):
        return self.operand.arity

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
    def arity(self):
        return self.operand.arity

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
    def arity(self):
        return self.operand.arity

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
    def arity(self):
        return self.operand.arity

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
    def arity(self):
        return self.operand.arity

    @cached_property
    def support(self):
        return self.operand.support
