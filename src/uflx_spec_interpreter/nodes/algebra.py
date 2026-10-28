from dataclasses import dataclass, field
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
    # The fields are the static type, so they must not take the defaults of AbstractExpression.
    value_shape: tuple = field()
    free_indices: tuple = field()
    argument_spaces: tuple
    opaque = ("value_shape", "free_indices", "argument_spaces")

    @cached_property
    def arity(self):
        return len(self.argument_spaces)


@dataclass(frozen=True, eq=False)
class Sum(AbstractExpression):
    identifier = "uflx.Sum"
    first: Base
    second: Base

    @cached_property
    def value_shape(self):
        return self.first.value_shape


@dataclass(frozen=True, eq=False)
class Product(AbstractExpression):
    identifier = "uflx.Product"
    first: Base
    second: Base

    @cached_property
    def arity(self):
        return self.first.arity + self.second.arity


@dataclass(frozen=True, eq=False)
class Division(AbstractExpression):
    identifier = "uflx.Division"
    first: Base
    second: Base


@dataclass(frozen=True, eq=False)
class Conjugate(AbstractExpression):
    identifier = "uflx.Conjugate"
    operand: Base

    @cached_property
    def value_shape(self):
        return self.operand.value_shape


@dataclass(frozen=True, eq=False)
class RealPart(AbstractExpression):
    identifier = "uflx.RealPart"
    operand: Base


@dataclass(frozen=True, eq=False)
class ImaginaryPart(AbstractExpression):
    identifier = "uflx.ImaginaryPart"
    operand: Base


@dataclass(frozen=True, eq=False)
class Absolute(AbstractExpression):
    identifier = "uflx.Absolute"
    operand: Base


@dataclass(frozen=True, eq=False)
class SquareRoot(AbstractExpression):
    identifier = "uflx.SquareRoot"
    operand: Base


@dataclass(frozen=True, eq=False)
class Logarithm(AbstractExpression):
    identifier = "uflx.Logarithm"
    operand: Base


@dataclass(frozen=True, eq=False)
class Inverse(AbstractExpression):
    identifier = "uflx.Inverse"
    operand: Base

    @cached_property
    def value_shape(self):
        return tuple(reversed(self.operand.value_shape))


@dataclass(frozen=True, eq=False)
class Power(AbstractExpression):
    identifier = "uflx.Power"
    first: Base
    second: Base


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
    expression: Base
    indices: tuple

    @cached_property
    def free_indices(self):
        # An Index takes the extent of its axis, and a gathered index states its own.
        out = dict(self.expression.free_indices)
        for k, i in enumerate(self.indices):
            if isinstance(i, Index):
                out[i] = self.expression.value_shape[k]
            elif not isinstance(i, int):
                out |= dict(i.free_indices)
        return tuple(out.items())


@dataclass(frozen=True, eq=False)
class ComponentTensor(AbstractExpression):
    identifier = "uflx.ComponentTensor"
    expression: Base
    indices: tuple
    scope = ("expression", "indices")

    def bound(self):
        return list(self.indices)

    @cached_property
    def value_shape(self):
        free = dict(self.expression.free_indices)
        return tuple(free[i] for i in self.indices)

    @cached_property
    def free_indices(self):
        return tuple((i, n) for i, n in self.expression.free_indices if i not in self.indices)


@dataclass(frozen=True, eq=False)
class IndexSum(AbstractExpression):
    identifier = "uflx.IndexSum"
    expression: Base
    indices: tuple
    scope = ("expression", "indices")

    def bound(self):
        return list(self.indices)

    @cached_property
    def free_indices(self):
        return tuple((i, n) for i, n in self.expression.free_indices if i not in self.indices)
