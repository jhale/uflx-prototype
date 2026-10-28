from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING, final

from .base import AbstractExpression

if TYPE_CHECKING:
    from .domains import AbstractDomain, CellMap, DecomposedDomain, DomainMap, SideMap
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


@final
class Coordinate(AbstractExpression):
    identifier = "uflx.Coordinate"
    domain: AbstractDomain
    opaque = ("domain",)

    @cached_property
    def value_shape(self):
        return (self.domain.ambient_coordinate_domain.dimension,)

    @cached_property
    def support(self):
        return self.domain


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
    def support(self):
        return self.domain


@final
class PullBack(AbstractExpression):
    identifier = "uflx.PullBack"
    expression: AbstractExpression
    domain: DomainMap | CellMap | SideMap

    @cached_property
    def value_shape(self):
        return self.expression.value_shape

    @cached_property
    def free_indices(self):
        return tuple((dict(self.expression.free_indices) | dict(self.domain.free_indices)).items())

    @cached_property
    def support(self):
        return self.domain.parameter_domain


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
    def support(self):
        return None


@final
class Normal(AbstractExpression):
    identifier = "uflx.Normal"
    ambient_domain: DecomposedDomain
    parameter_domain: DecomposedDomain
    side: SideMap

    @cached_property
    def value_shape(self):
        return (self.ambient_domain.ambient_coordinate_domain.dimension,)

    @cached_property
    def support(self):
        return self.parameter_domain
