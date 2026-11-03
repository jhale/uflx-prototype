from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING, final

from .base import AbstractExpression
from .scalar_types import ScalarType

if TYPE_CHECKING:
    pass


@final
class Sum(AbstractExpression):
    matching_domains = True
    value_fields = ("first", "second")
    identifier = "uflx.Sum"
    commuting = ("first", "second")
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
    def domain(self):
        return self.second.domain or self.first.domain

    def _validate(self):
        _validate_scalar_arithmetic(self)


@final
class Product(AbstractExpression):
    matching_domains = True
    value_fields = ("first", "second")
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
    def domain(self):
        return self.second.domain or self.first.domain

    def _validate(self):
        _validate_scalar_arithmetic(self)


@final
class Division(AbstractExpression):
    matching_domains = True
    value_fields = ("first", "second")
    identifier = "uflx.Division"
    commuting = ("first",)
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
    def domain(self):
        return self.second.domain or self.first.domain

    def _validate(self):
        _validate_scalar_arithmetic(self)


@final
class Power(AbstractExpression):
    matching_domains = True
    value_fields = ("first", "second")
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
    def domain(self):
        return self.second.domain or self.first.domain

    def _validate(self):
        _validate_scalar_arithmetic(self)


@final
class Conjugate(AbstractExpression):
    value_fields = ("operand",)
    identifier = "uflx.Conjugate"
    commuting = ("operand",)
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
    def domain(self):
        return self.operand.domain

    def _validate(self):
        from ..validation import typed

        typed(self.operand, AbstractExpression, "operand must be an expression")


@final
class RealPart(AbstractExpression):
    value_fields = ("operand",)
    identifier = "uflx.RealPart"
    commuting = ("operand",)
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
    def domain(self):
        return self.operand.domain

    def _validate(self):
        from ..validation import scalar_operand

        scalar_operand(self.operand)


@final
class ImaginaryPart(AbstractExpression):
    value_fields = ("operand",)
    identifier = "uflx.ImaginaryPart"
    commuting = ("operand",)
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
    def domain(self):
        return self.operand.domain

    def _validate(self):
        from ..validation import scalar_operand

        scalar_operand(self.operand)


@final
class Absolute(AbstractExpression):
    value_fields = ("operand",)
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
    def domain(self):
        return self.operand.domain

    def _validate(self):
        from ..validation import scalar_operand

        scalar_operand(self.operand)


@final
class SquareRoot(AbstractExpression):
    value_fields = ("operand",)
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
    def domain(self):
        return self.operand.domain

    def _validate(self):
        from ..validation import scalar_operand

        scalar_operand(self.operand)


@final
class Logarithm(AbstractExpression):
    value_fields = ("operand",)
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
    def domain(self):
        return self.operand.domain

    def _validate(self):
        from ..validation import scalar_operand

        scalar_operand(self.operand)


def _validate_scalar_arithmetic(e):
    from ..validation import require, typed

    for f in (e.first, e.second):
        typed(f, AbstractExpression, "arithmetic operands must be expressions")
        require(f.value_shape == (), "scalar arithmetic requires scalar operands")
    if isinstance(e, Sum):
        require(e.first.argument_spaces == e.second.argument_spaces, "sum argument spaces differ")
    if isinstance(e, (Division, Power)):
        require(not e.second.argument_spaces, "nonlinear operand has argument spaces")
    if isinstance(e, Power):
        require(not e.first.argument_spaces, "power base has argument spaces")
