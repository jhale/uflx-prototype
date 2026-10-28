from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING, final

from .base import AbstractExpression

if TYPE_CHECKING:
    from .domains import AbstractDomain, DomainMap, SideMap
    from .forms import Function


@final
class DirectionalDerivative(AbstractExpression):
    identifier = "uflx.DirectionalDerivative"
    expression: AbstractExpression
    variable: Function
    point: AbstractExpression
    direction: AbstractExpression
    scope = ("expression", "variable")

    def bound(self):
        return [self.variable]

    @cached_property
    def value_shape(self):
        return self.expression.value_shape

    @cached_property
    def free_indices(self):
        return tuple(
            dict(
                self.expression.free_indices + self.point.free_indices + self.direction.free_indices
            ).items()
        )

    @cached_property
    def argument_spaces(self):
        return self.expression.argument_spaces

    @cached_property
    def scalar_type(self):
        return self.expression.scalar_type

    @cached_property
    def support(self):
        return self.direction.support or self.point.support or self.expression.support


@final
class Coordinate(AbstractExpression):
    identifier = "uflx.Coordinate"
    domain: AbstractDomain
    opaque = ("domain",)

    @cached_property
    def value_shape(self):
        return (self.domain.ambient_coordinate_domain.dimension,)

    @cached_property
    def scalar_type(self):
        return "real"

    @cached_property
    def support(self):
        return self.domain

    @cached_property
    def free_indices(self):
        return ()

    @cached_property
    def argument_spaces(self):
        return ()


@final
class Gradient(AbstractExpression):
    identifier = "uflx.Gradient"
    expression: AbstractExpression
    domain: AbstractDomain
    opaque = ("domain",)

    @cached_property
    def value_shape(self):
        return (*self.expression.value_shape, self.domain.ambient_coordinate_domain.dimension)

    @cached_property
    def scalar_type(self):
        return self.joined_scalar_type(self.expression)

    @cached_property
    def support(self):
        return self.domain

    @cached_property
    def free_indices(self):
        return self.expression.free_indices

    @cached_property
    def argument_spaces(self):
        return self.expression.argument_spaces


@final
class PullBack(AbstractExpression):
    identifier = "uflx.PullBack"
    expression: AbstractExpression
    domain: DomainMap | SideMap

    @cached_property
    def value_shape(self):
        return self.expression.value_shape

    @cached_property
    def free_indices(self):
        return tuple((dict(self.expression.free_indices) | dict(self.domain.free_indices)).items())

    @cached_property
    def scalar_type(self):
        return self.joined_scalar_type(self.expression)

    @cached_property
    def support(self):
        return self.domain.parameter_domain

    @cached_property
    def argument_spaces(self):
        return self.expression.argument_spaces


@final
class Integral(AbstractExpression):
    identifier = "uflx.Integral"
    expression: AbstractExpression
    domain: AbstractDomain
    opaque = ("domain",)

    @cached_property
    def value_shape(self):
        return self.expression.value_shape

    @cached_property
    def free_indices(self):
        # The free indices of the pieces of P are summed over.
        bound = {i for piece in getattr(self.domain, "pieces", ()) for i, _ in piece.free_indices}
        return tuple((i, n) for i, n in self.expression.free_indices if i not in bound)

    @cached_property
    def scalar_type(self):
        return self.joined_scalar_type(self.expression)

    @cached_property
    def support(self):
        return None

    @cached_property
    def argument_spaces(self):
        return self.expression.argument_spaces


@final
class Normal(AbstractExpression):
    identifier = "uflx.Normal"
    side_map: SideMap

    @cached_property
    def value_shape(self):
        return (self.side_map.ambient_domain.ambient_coordinate_domain.dimension,)

    @cached_property
    def scalar_type(self):
        return "real"

    @cached_property
    def support(self):
        return self.side_map.parameter_domain

    @cached_property
    def free_indices(self):
        return ()

    @cached_property
    def argument_spaces(self):
        return ()
