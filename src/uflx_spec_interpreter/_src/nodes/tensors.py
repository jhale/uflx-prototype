from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING, final

from .base import AbstractExpression
from .indices import GatheredIndex, Index

if TYPE_CHECKING:
    from .indices import GatheredIndex, Index


@final
class Tensor(AbstractExpression):
    matching_domains = True
    value_fields = ("entries",)
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

    def _validate(self):
        from ..validation import require, tuple_of

        tuple_of(self.entries, AbstractExpression)
        require(bool(self.entries), "tensor entries must be nonempty")
        require(
            all(
                f.value_shape == self.entries[0].value_shape
                and f.argument_spaces == self.entries[0].argument_spaces
                for f in self.entries
            ),
            "tensor entry types differ",
        )


@final
class Indexed(AbstractExpression):
    index_fields = ("indices",)
    value_fields = ("expression",)
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

    def _validate(self):
        from ..validation import entries, merge_indices, require, typed

        typed(self.expression, AbstractExpression, "indexed value must be an expression")
        require(bool(self.indices), "Indexed requires nonempty entries")
        merge_indices(
            self.expression.free_indices, entries(self.expression.value_shape, self.indices)
        )


@final
class ComponentTensor(AbstractExpression):
    value_fields = ("expression",)
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

    def _validate(self):
        _validate_index_binding(self)


@final
class IndexSum(AbstractExpression):
    value_fields = ("expression",)
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

    def _validate(self):
        _validate_index_binding(self)


@final
class Inverse(AbstractExpression):
    value_fields = ("operand",)
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

    def _validate(self):
        from ..validation import matrix_operand

        matrix_operand(self.operand, rectangular=True)


@final
class Determinant(AbstractExpression):
    value_fields = ("operand",)
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

    def _validate(self):
        from ..validation import matrix_operand

        matrix_operand(self.operand)


def _validate_index_binding(e):
    from ..validation import require, tuple_of, typed

    typed(e.expression, AbstractExpression, "binding requires an expression")
    tuple_of(e.indices, Index)
    require(
        bool(e.indices) and len(set(e.indices)) == len(e.indices),
        "bound indices must be nonempty and distinct",
    )
    require(e.expression.value_shape == (), "index binding requires a scalar")
    require(all(i in dict(e.expression.free_indices) for i in e.indices), "bound index is not free")
