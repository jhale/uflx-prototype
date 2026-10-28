from dataclasses import dataclass

from .base import AbstractExpression, Base


@dataclass(frozen=True, eq=False)
class D(AbstractExpression):
    identifier = "uflx.DirectionalDerivative"
    f: Base
    x: Base
    a: Base
    h: Base
    scope = ("f", "x")

    def bound(self):
        return [self.x]


@dataclass(frozen=True, eq=False)
class Coordinate(AbstractExpression):
    identifier = "uflx.Coordinate"
    P: Base
    opaque = ("P",)


@dataclass(frozen=True, eq=False)
class Gradient(AbstractExpression):
    identifier = "uflx.Gradient"
    b: Base
    P: Base
    opaque = ("P",)


@dataclass(frozen=True, eq=False)
class PullBack(AbstractExpression):
    identifier = "uflx.PullBack"
    b: Base
    m: Base


@dataclass(frozen=True, eq=False)
class Integral(AbstractExpression):
    identifier = "uflx.Integral"
    b: Base
    P: Base
    opaque = ("P",)


@dataclass(frozen=True, eq=False)
class Normal(AbstractExpression):
    identifier = "uflx.Normal"
    D: Base
    E: Base
    side: Base
