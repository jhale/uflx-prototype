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
    commuting = ("entries",)
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
    def scalar_type(self):
        return self.joined_scalar_type(*self.entries)

    @cached_property
    def domain(self):
        return next((x.domain for x in reversed(self.entries) if x.domain), None)


@final
class Indexed(AbstractExpression):
    identifier = "uflx.Indexed"
    commuting = ("expression",)
    expression: AbstractExpression
    indices: tuple[int | Index | GatheredIndex | ...]

    @cached_property
    def free_indices(self):
        from ..validation import entries, merge_indices

        return merge_indices(
            self.expression.free_indices, entries(self.expression.value_shape, self.indices)
        )

    @cached_property
    def value_shape(self):
        return ()

    @cached_property
    def argument_spaces(self):
        return self.expression.argument_spaces

    @cached_property
    def scalar_type(self):
        return self.joined_scalar_type(self.expression)

    @cached_property
    def domain(self):
        return self.expression.domain


@final
class ComponentTensor(AbstractExpression):
    identifier = "uflx.ComponentTensor"
    commuting = ("expression",)
    expression: AbstractExpression
    indices: tuple[Index | ...]
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
    def argument_spaces(self):
        return self.expression.argument_spaces

    @cached_property
    def scalar_type(self):
        return self.joined_scalar_type(self.expression)

    @cached_property
    def domain(self):
        return self.expression.domain


@final
class IndexSum(AbstractExpression):
    identifier = "uflx.IndexSum"
    commuting = ("expression",)
    expression: AbstractExpression
    indices: tuple[Index | ...]
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
    def scalar_type(self):
        return self.joined_scalar_type(self.expression)

    @cached_property
    def domain(self):
        return self.expression.domain


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
    def scalar_type(self):
        return self.joined_scalar_type(self.operand)

    @cached_property
    def domain(self):
        return self.operand.domain


@final
class Determinant(AbstractExpression):
    identifier = "uflx.Determinant"
    operand: AbstractExpression

    @cached_property
    def value_shape(self):
        return ()

    @cached_property
    def free_indices(self):
        return self.operand.free_indices

    @cached_property
    def argument_spaces(self):
        return ()

    @cached_property
    def scalar_type(self):
        return self.joined_scalar_type(self.operand)

    @cached_property
    def domain(self):
        return self.operand.domain
