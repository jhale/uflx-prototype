from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING, final

from .base import AbstractExpression, Base, Index

if TYPE_CHECKING:
    from .forms import Space


@final
class Num(AbstractExpression):
    identifier = "toy.Num"
    value: int | float | complex
    opaque = ("value",)


@final
class Zero(AbstractExpression):
    identifier = "uflx.Zero"
    # The fields are the static type, so they must not take the defaults of AbstractExpression.
    value_shape: tuple[int, ...]
    free_indices: tuple[tuple[Index, int], ...]
    argument_spaces: tuple[Space, ...]
    opaque = ("value_shape", "free_indices", "argument_spaces")

    @cached_property
    def arity(self):
        return len(self.argument_spaces)


@final
class Sum(AbstractExpression):
    identifier = "uflx.Sum"
    first: AbstractExpression
    second: AbstractExpression

    @cached_property
    def value_shape(self):
        return self.first.value_shape


@final
class Product(AbstractExpression):
    identifier = "uflx.Product"
    first: AbstractExpression
    second: AbstractExpression

    @cached_property
    def arity(self):
        return self.first.arity + self.second.arity


@final
class Division(AbstractExpression):
    identifier = "uflx.Division"
    first: AbstractExpression
    second: AbstractExpression


@final
class Conjugate(AbstractExpression):
    identifier = "uflx.Conjugate"
    operand: AbstractExpression

    @cached_property
    def value_shape(self):
        return self.operand.value_shape


@final
class RealPart(AbstractExpression):
    identifier = "uflx.RealPart"
    operand: AbstractExpression


@final
class ImaginaryPart(AbstractExpression):
    identifier = "uflx.ImaginaryPart"
    operand: AbstractExpression


@final
class Absolute(AbstractExpression):
    identifier = "uflx.Absolute"
    operand: AbstractExpression


@final
class SquareRoot(AbstractExpression):
    identifier = "uflx.SquareRoot"
    operand: AbstractExpression


@final
class Logarithm(AbstractExpression):
    identifier = "uflx.Logarithm"
    operand: AbstractExpression


@final
class Inverse(AbstractExpression):
    identifier = "uflx.Inverse"
    operand: AbstractExpression

    @cached_property
    def value_shape(self):
        return tuple(reversed(self.operand.value_shape))


@final
class Power(AbstractExpression):
    identifier = "uflx.Power"
    first: AbstractExpression
    second: AbstractExpression


@final
class Tensor(AbstractExpression):
    identifier = "uflx.Tensor"
    entries: tuple[AbstractExpression, ...]

    @cached_property
    def value_shape(self):
        return (len(self.entries), *self.entries[0].value_shape)


@final
class Indexed(AbstractExpression):
    identifier = "uflx.Indexed"
    expression: AbstractExpression
    indices: tuple[int | Index | GatheredIndex, ...]

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


@final
class ComponentTensor(AbstractExpression):
    identifier = "uflx.ComponentTensor"
    expression: AbstractExpression
    indices: tuple[Index, ...]
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


@final
class IndexSum(AbstractExpression):
    identifier = "uflx.IndexSum"
    expression: AbstractExpression
    indices: tuple[Index, ...]
    scope = ("expression", "indices")

    def bound(self):
        return list(self.indices)

    @cached_property
    def free_indices(self):
        return tuple((i, n) for i, n in self.expression.free_indices if i not in self.indices)


@final
class GatheredIndex(Base):
    identifier = "uflx.GatheredIndex"
    # An index read from a table at the given entries, an int or an Index for each axis.
    table: AbstractExpression
    indices: tuple[int | Index, ...]

    @cached_property
    def free_indices(self):
        # Each Index entry takes the extent of its axis of the table.
        shape = self.table.value_shape
        return tuple((j, shape[m]) for m, j in enumerate(self.indices) if not isinstance(j, int))
