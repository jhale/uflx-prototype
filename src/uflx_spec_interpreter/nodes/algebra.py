from __future__ import annotations

import math
from abc import abstractmethod
from functools import cached_property
from typing import TYPE_CHECKING, final

from .base import AbstractExpression, Base, Index

if TYPE_CHECKING:
    from .forms import Space


@final
class Integer(AbstractExpression):
    identifier = "uflx.Integer"
    value: int
    opaque = ("value",)

    def __new__(cls, value):
        # An exact int, as rule 1 asks, so not a bool.
        if type(value) is not int:
            raise TypeError(f"Integer takes an int, not {value!r}")
        return super().__new__(cls, value)

    @cached_property
    def value_shape(self):
        return ()

    @cached_property
    def free_indices(self):
        return ()

    @cached_property
    def arity(self):
        return 0

    @cached_property
    def support(self):
        return None


@final
class Float(AbstractExpression):
    identifier = "uflx.Float"
    value: float
    opaque = ("value",)

    def __new__(cls, value):
        # An exact finite float, as rule 1 asks.
        if type(value) is not float:
            raise TypeError(f"Float takes a float, not {value!r}")
        if not math.isfinite(value):
            raise ValueError(f"Float takes a finite float, not {value!r}")
        return super().__new__(cls, value)

    @cached_property
    def value_shape(self):
        return ()

    @cached_property
    def free_indices(self):
        return ()

    @cached_property
    def arity(self):
        return 0

    @cached_property
    def support(self):
        return None


@final
class ImaginaryUnit(AbstractExpression):
    identifier = "uflx.ImaginaryUnit"

    @cached_property
    def value_shape(self):
        return ()

    @cached_property
    def free_indices(self):
        return ()

    @cached_property
    def arity(self):
        return 0

    @cached_property
    def support(self):
        return None


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

    @cached_property
    def support(self):
        return None


@final
class Sum(AbstractExpression):
    identifier = "uflx.Sum"
    first: AbstractExpression
    second: AbstractExpression

    @cached_property
    def value_shape(self):
        return self.first.value_shape

    @cached_property
    def free_indices(self):
        return tuple(dict(self.first.free_indices + self.second.free_indices).items())

    @cached_property
    def arity(self):
        return self.first.arity

    @cached_property
    def support(self):
        return self.second.support or self.first.support


@final
class Product(AbstractExpression):
    identifier = "uflx.Product"
    first: AbstractExpression
    second: AbstractExpression

    @cached_property
    def arity(self):
        return self.first.arity + self.second.arity

    @cached_property
    def value_shape(self):
        return ()

    @cached_property
    def free_indices(self):
        return tuple(dict(self.first.free_indices + self.second.free_indices).items())

    @cached_property
    def support(self):
        return self.second.support or self.first.support


@final
class Division(AbstractExpression):
    identifier = "uflx.Division"
    first: AbstractExpression
    second: AbstractExpression

    @cached_property
    def value_shape(self):
        return ()

    @cached_property
    def free_indices(self):
        return tuple(dict(self.first.free_indices + self.second.free_indices).items())

    @cached_property
    def arity(self):
        return self.first.arity

    @cached_property
    def support(self):
        return self.second.support or self.first.support


@final
class Conjugate(AbstractExpression):
    identifier = "uflx.Conjugate"
    operand: AbstractExpression

    @cached_property
    def value_shape(self):
        return self.operand.value_shape

    @cached_property
    def free_indices(self):
        return self.operand.free_indices

    @cached_property
    def arity(self):
        return self.operand.arity

    @cached_property
    def support(self):
        return self.operand.support


@final
class RealPart(AbstractExpression):
    identifier = "uflx.RealPart"
    operand: AbstractExpression

    @cached_property
    def value_shape(self):
        return ()

    @cached_property
    def free_indices(self):
        return self.operand.free_indices

    @cached_property
    def arity(self):
        return self.operand.arity

    @cached_property
    def support(self):
        return self.operand.support


@final
class ImaginaryPart(AbstractExpression):
    identifier = "uflx.ImaginaryPart"
    operand: AbstractExpression

    @cached_property
    def value_shape(self):
        return ()

    @cached_property
    def free_indices(self):
        return self.operand.free_indices

    @cached_property
    def arity(self):
        return self.operand.arity

    @cached_property
    def support(self):
        return self.operand.support


@final
class Absolute(AbstractExpression):
    identifier = "uflx.Absolute"
    operand: AbstractExpression

    @cached_property
    def value_shape(self):
        return ()

    @cached_property
    def free_indices(self):
        return self.operand.free_indices

    @cached_property
    def arity(self):
        return self.operand.arity

    @cached_property
    def support(self):
        return self.operand.support


@final
class SquareRoot(AbstractExpression):
    identifier = "uflx.SquareRoot"
    operand: AbstractExpression

    @cached_property
    def value_shape(self):
        return ()

    @cached_property
    def free_indices(self):
        return self.operand.free_indices

    @cached_property
    def arity(self):
        return self.operand.arity

    @cached_property
    def support(self):
        return self.operand.support


@final
class Logarithm(AbstractExpression):
    identifier = "uflx.Logarithm"
    operand: AbstractExpression

    @cached_property
    def value_shape(self):
        return ()

    @cached_property
    def free_indices(self):
        return self.operand.free_indices

    @cached_property
    def arity(self):
        return self.operand.arity

    @cached_property
    def support(self):
        return self.operand.support


@final
class Inverse(AbstractExpression):
    identifier = "uflx.Inverse"
    operand: AbstractExpression

    @cached_property
    def value_shape(self):
        return tuple(reversed(self.operand.value_shape))

    @cached_property
    def free_indices(self):
        return self.operand.free_indices

    @cached_property
    def arity(self):
        return self.operand.arity

    @cached_property
    def support(self):
        return self.operand.support


@final
class Power(AbstractExpression):
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
    def arity(self):
        return self.first.arity

    @cached_property
    def support(self):
        return self.second.support or self.first.support


@final
class Tensor(AbstractExpression):
    identifier = "uflx.Tensor"
    entries: tuple[AbstractExpression, ...]

    @cached_property
    def value_shape(self):
        return (len(self.entries), *self.entries[0].value_shape)

    @cached_property
    def free_indices(self):
        return tuple(dict(p for x in self.entries for p in x.free_indices).items())

    @cached_property
    def arity(self):
        return self.entries[0].arity

    @cached_property
    def support(self):
        return next((x.support for x in reversed(self.entries) if x.support), None)


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

    @cached_property
    def value_shape(self):
        return ()

    @cached_property
    def arity(self):
        return self.expression.arity

    @cached_property
    def support(self):
        return self.expression.support


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

    @cached_property
    def arity(self):
        return self.expression.arity

    @cached_property
    def support(self):
        return self.expression.support


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

    @cached_property
    def value_shape(self):
        return ()

    @cached_property
    def arity(self):
        return self.expression.arity

    @cached_property
    def support(self):
        return self.expression.support


class AbstractIndexValue(Base):
    # A table of index values that the solver gives, which is not an expression.
    @property
    @abstractmethod
    def extent(self) -> int: ...

    @property
    @abstractmethod
    def shape(self) -> tuple: ...


@final
class IndexTable(AbstractIndexValue):
    identifier = "toy.IndexTable"
    # The toy knows the values, which the specification leaves to a library.
    values: tuple
    extent: int
    opaque = ("values", "extent")

    @cached_property
    def shape(self):
        shape, x = [], self.values
        while isinstance(x, tuple):
            shape.append(len(x))
            x = x[0]
        return tuple(shape)


@final
class GatheredIndex(Base):
    identifier = "uflx.GatheredIndex"
    # An index read from a table at the given entries, an int or an Index for each axis.
    table: AbstractIndexValue
    indices: tuple[int | Index | GatheredIndex, ...]

    @property
    def extent(self):
        return self.table.extent

    @cached_property
    def free_indices(self):
        # Each Index entry takes the extent of its axis of the table, and a nested gather its own.
        out = {}
        for m, j in enumerate(self.indices):
            if isinstance(j, Index):
                out[j] = self.table.shape[m]
            elif isinstance(j, GatheredIndex):
                out |= dict(j.free_indices)
        return tuple(out.items())
