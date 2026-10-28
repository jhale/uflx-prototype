from dataclasses import dataclass
from functools import cached_property

from .base import AbstractExpression, Base, Index


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

    @cached_property
    def value_shape(self):
        return self.shape

    @cached_property
    def free_indices(self):
        return tuple(self.free)

    @cached_property
    def arity(self):
        return len(self.args)


@dataclass(frozen=True, eq=False)
class Sum(AbstractExpression):
    identifier = "uflx.Sum"
    a: Base
    b: Base

    @cached_property
    def value_shape(self):
        return self.a.value_shape


@dataclass(frozen=True, eq=False)
class Product(AbstractExpression):
    identifier = "uflx.Product"
    a: Base
    b: Base

    @cached_property
    def arity(self):
        return self.a.arity + self.b.arity


@dataclass(frozen=True, eq=False)
class Division(AbstractExpression):
    identifier = "uflx.Division"
    a: Base
    c: Base


@dataclass(frozen=True, eq=False)
class Conjugate(AbstractExpression):
    identifier = "uflx.Conjugate"
    a: Base

    @cached_property
    def value_shape(self):
        return self.a.value_shape


@dataclass(frozen=True, eq=False)
class RealPart(AbstractExpression):
    identifier = "uflx.RealPart"
    a: Base


@dataclass(frozen=True, eq=False)
class ImaginaryPart(AbstractExpression):
    identifier = "uflx.ImaginaryPart"
    a: Base


@dataclass(frozen=True, eq=False)
class Absolute(AbstractExpression):
    identifier = "uflx.Absolute"
    a: Base


@dataclass(frozen=True, eq=False)
class SquareRoot(AbstractExpression):
    identifier = "uflx.SquareRoot"
    a: Base


@dataclass(frozen=True, eq=False)
class Logarithm(AbstractExpression):
    identifier = "uflx.Logarithm"
    a: Base


@dataclass(frozen=True, eq=False)
class Inverse(AbstractExpression):
    identifier = "uflx.Inverse"
    a: Base

    @cached_property
    def value_shape(self):
        return tuple(reversed(self.a.value_shape))


@dataclass(frozen=True, eq=False)
class Power(AbstractExpression):
    identifier = "uflx.Power"
    a: Base
    c: Base


@dataclass(frozen=True, eq=False)
class Tensor(AbstractExpression):
    identifier = "uflx.Tensor"
    entries: tuple

    @cached_property
    def value_shape(self):
        return (len(self.entries), *self.entries[0].value_shape)


@dataclass(frozen=True, eq=False)
class Indexed(AbstractExpression):
    identifier = "uflx.Indexed"
    e: Base
    idx: tuple

    @cached_property
    def free_indices(self):
        # An Index takes the extent of its axis, and a gathered index states its own.
        out = dict(self.e.free_indices)
        for k, i in enumerate(self.idx):
            if isinstance(i, Index):
                out[i] = self.e.value_shape[k]
            elif not isinstance(i, int):
                out |= dict(i.free_indices)
        return tuple(out.items())


@dataclass(frozen=True, eq=False)
class ComponentTensor(AbstractExpression):
    identifier = "uflx.ComponentTensor"
    e: Base
    idx: tuple
    scope = ("e", "idx")

    def bound(self):
        return list(self.idx)

    @cached_property
    def value_shape(self):
        free = dict(self.e.free_indices)
        return tuple(free[i] for i in self.idx)

    @cached_property
    def free_indices(self):
        return tuple((i, n) for i, n in self.e.free_indices if i not in self.idx)


@dataclass(frozen=True, eq=False)
class IndexSum(AbstractExpression):
    identifier = "uflx.IndexSum"
    e: Base
    idx: tuple
    scope = ("e", "idx")

    def bound(self):
        return list(self.idx)

    @cached_property
    def free_indices(self):
        return tuple((i, n) for i, n in self.e.free_indices if i not in self.idx)
