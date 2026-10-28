from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING, final

from .base import AbstractExpression
from .indices import EllipsisIndex, GatheredIndex, Index

if TYPE_CHECKING:
    from .indices import EllipsisIndex, GatheredIndex, Index


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
    def argument_spaces(self):
        return self.entries[0].argument_spaces

    @cached_property
    def support(self):
        return next((x.support for x in reversed(self.entries) if x.support), None)


@final
class Indexed(AbstractExpression):
    identifier = "uflx.Indexed"
    expression: AbstractExpression
    indices: tuple[int | Index | GatheredIndex | EllipsisIndex, ...]

    @cached_property
    def free_indices(self):
        # An Index takes the extent of its axis, an EllipsisIndex the extents of every axis that
        # the other entries leave, and a gathered index states its own.
        shape = self.expression.value_shape
        covered = len(shape) - len(self.indices) + 1
        out = dict(self.expression.free_indices)
        k = 0
        for i in self.indices:
            if isinstance(i, EllipsisIndex):
                out[i] = tuple(shape[k : k + covered])
                k += covered
                continue
            if isinstance(i, Index):
                out[i] = shape[k]
            elif isinstance(i, GatheredIndex):
                out |= dict(i.free_indices)
            k += 1
        return tuple(out.items())

    @cached_property
    def value_shape(self):
        return ()

    @cached_property
    def argument_spaces(self):
        return self.expression.argument_spaces

    @cached_property
    def support(self):
        return self.expression.support


@final
class ComponentTensor(AbstractExpression):
    identifier = "uflx.ComponentTensor"
    expression: AbstractExpression
    indices: tuple[Index | EllipsisIndex, ...]
    scope = ("expression", "indices")

    def bound(self):
        return list(self.indices)

    @cached_property
    def value_shape(self):
        # An EllipsisIndex gives one axis for each of its extents.
        free = dict(self.expression.free_indices)
        return tuple(
            n
            for i in self.indices
            for n in (free[i] if isinstance(i, EllipsisIndex) else (free[i],))
        )

    @cached_property
    def free_indices(self):
        return tuple((i, n) for i, n in self.expression.free_indices if i not in self.indices)

    @cached_property
    def argument_spaces(self):
        return self.expression.argument_spaces

    @cached_property
    def support(self):
        return self.expression.support


@final
class IndexSum(AbstractExpression):
    identifier = "uflx.IndexSum"
    expression: AbstractExpression
    indices: tuple[Index | EllipsisIndex, ...]
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
    def argument_spaces(self):
        return self.expression.argument_spaces

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
    def argument_spaces(self):
        return ()

    @cached_property
    def support(self):
        return self.operand.support
