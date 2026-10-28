from dataclasses import dataclass
from functools import cached_property

from .base import AbstractExpression, Base


@dataclass(frozen=True, eq=False)
class DirectionalDerivative(AbstractExpression):
    identifier = "uflx.DirectionalDerivative"
    f: Base
    x: Base
    a: Base
    h: Base
    scope = ("f", "x")

    def bound(self):
        return [self.x]

    @cached_property
    def value_shape(self):
        return self.f.value_shape


@dataclass(frozen=True, eq=False)
class Coordinate(AbstractExpression):
    identifier = "uflx.Coordinate"
    P: Base
    opaque = ("P",)

    @cached_property
    def value_shape(self):
        return (self.P.ambient_dim,)

    @cached_property
    def support(self):
        return self.P


@dataclass(frozen=True, eq=False)
class Gradient(AbstractExpression):
    identifier = "uflx.Gradient"
    b: Base
    P: Base
    opaque = ("P",)

    @cached_property
    def value_shape(self):
        return (*self.b.value_shape, self.P.ambient_dim)

    @cached_property
    def support(self):
        return self.P


@dataclass(frozen=True, eq=False)
class PullBack(AbstractExpression):
    identifier = "uflx.PullBack"
    b: Base
    m: Base

    @cached_property
    def value_shape(self):
        return self.b.value_shape

    @cached_property
    def free_indices(self):
        return tuple((dict(self.b.free_indices) | dict(self.m.free_indices)).items())

    @cached_property
    def support(self):
        return self.m.parameter_domain


@dataclass(frozen=True, eq=False)
class Integral(AbstractExpression):
    identifier = "uflx.Integral"
    b: Base
    P: Base
    opaque = ("P",)

    @cached_property
    def value_shape(self):
        return self.b.value_shape

    @cached_property
    def free_indices(self):
        # The free indices of the pieces of P are summed over.
        bound = {i for piece in getattr(self.P, "pieces", ()) for i, _ in piece.free_indices}
        return tuple((i, n) for i, n in self.b.free_indices if i not in bound)

    @cached_property
    def support(self):
        return None


@dataclass(frozen=True, eq=False)
class Normal(AbstractExpression):
    identifier = "uflx.Normal"
    D: Base
    E: Base
    side: Base

    @cached_property
    def value_shape(self):
        return (self.D.ambient_dim,)

    @cached_property
    def support(self):
        return self.E
