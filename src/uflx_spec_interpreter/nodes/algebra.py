from dataclasses import dataclass

from .base import AbstractExpression, Base


@dataclass(frozen=True, eq=False)
class Num(AbstractExpression):
    identifier = "toy.Num"
    value: complex
    opaque = ("value",)


@dataclass(frozen=True, eq=False)
class Zero(AbstractExpression):
    identifier = "uflx.Zero"
    shape: tuple
    free: tuple
    args: tuple
    opaque = ("shape", "free", "args")


@dataclass(frozen=True, eq=False)
class Sum(AbstractExpression):
    identifier = "uflx.Sum"
    a: Base
    b: Base


@dataclass(frozen=True, eq=False)
class Prod(AbstractExpression):
    identifier = "uflx.Product"
    a: Base
    b: Base


@dataclass(frozen=True, eq=False)
class Div(AbstractExpression):
    identifier = "uflx.Division"
    a: Base
    c: Base


@dataclass(frozen=True, eq=False)
class Conj(AbstractExpression):
    identifier = "uflx.Conjugate"
    a: Base


@dataclass(frozen=True, eq=False)
class Re(AbstractExpression):
    identifier = "uflx.RealPart"
    a: Base


@dataclass(frozen=True, eq=False)
class Im(AbstractExpression):
    identifier = "uflx.ImaginaryPart"
    a: Base


@dataclass(frozen=True, eq=False)
class Abs(AbstractExpression):
    identifier = "uflx.Absolute"
    a: Base


@dataclass(frozen=True, eq=False)
class Sqrt(AbstractExpression):
    identifier = "uflx.SquareRoot"
    a: Base


@dataclass(frozen=True, eq=False)
class Log(AbstractExpression):
    identifier = "uflx.Logarithm"
    a: Base


@dataclass(frozen=True, eq=False)
class Inv(AbstractExpression):
    identifier = "uflx.Inverse"
    a: Base


@dataclass(frozen=True, eq=False)
class Power(AbstractExpression):
    identifier = "uflx.Power"
    a: Base
    c: Base


@dataclass(frozen=True, eq=False)
class Tensor(AbstractExpression):
    identifier = "uflx.Tensor"
    entries: tuple


@dataclass(frozen=True, eq=False)
class Indexed(AbstractExpression):
    identifier = "uflx.Indexed"
    e: Base
    idx: tuple


@dataclass(frozen=True, eq=False)
class CT(AbstractExpression):
    identifier = "uflx.ComponentTensor"
    e: Base
    idx: tuple
    scope = ("e", "idx")

    def bound(self):
        return list(self.idx)


@dataclass(frozen=True, eq=False)
class ISum(AbstractExpression):
    identifier = "uflx.IndexSum"
    e: Base
    idx: tuple
    scope = ("e", "idx")

    def bound(self):
        return list(self.idx)
