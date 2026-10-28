from dataclasses import dataclass
from functools import cached_property

from .base import AbstractExpression, Base


@dataclass(frozen=True, eq=False)
class DirectionalDerivative(AbstractExpression):
    identifier = "uflx.DirectionalDerivative"
    expression: Base
    variable: Base
    point: Base
    direction: Base
    scope = ("expression", "variable")

    def bound(self):
        return [self.variable]

    @cached_property
    def value_shape(self):
        return self.expression.value_shape


@dataclass(frozen=True, eq=False)
class Coordinate(AbstractExpression):
    identifier = "uflx.Coordinate"
    domain: Base
    opaque = ("domain",)

    @cached_property
    def value_shape(self):
        return (self.domain.ambient_dim,)

    @cached_property
    def support(self):
        return self.domain


@dataclass(frozen=True, eq=False)
class Gradient(AbstractExpression):
    identifier = "uflx.Gradient"
    expression: Base
    domain: Base
    opaque = ("domain",)

    @cached_property
    def value_shape(self):
        return (*self.expression.value_shape, self.domain.ambient_dim)

    @cached_property
    def support(self):
        return self.domain


@dataclass(frozen=True, eq=False)
class PullBack(AbstractExpression):
    identifier = "uflx.PullBack"
    expression: Base
    domain: Base

    @cached_property
    def value_shape(self):
        return self.expression.value_shape

    @cached_property
    def free_indices(self):
        return tuple((dict(self.expression.free_indices) | dict(self.domain.free_indices)).items())

    @cached_property
    def support(self):
        return self.domain.parameter_domain


@dataclass(frozen=True, eq=False)
class Integral(AbstractExpression):
    identifier = "uflx.Integral"
    expression: Base
    domain: Base
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


@dataclass(frozen=True, eq=False)
class Normal(AbstractExpression):
    identifier = "uflx.Normal"
    ambient_domain: Base
    parameter_domain: Base
    side: Base

    @cached_property
    def value_shape(self):
        return (self.ambient_domain.ambient_dim,)

    @cached_property
    def support(self):
        return self.parameter_domain
