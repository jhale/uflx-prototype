from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING, final

from .base import AbstractExpression
from .indices import GatheredIndex, Index

if TYPE_CHECKING:
    from .indices import GatheredIndex, Index


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
