# Copyright (C) 2026 Jack S. Hale
#
# This file is part of uflx-core-experimental (https://fenics.github.io/uflx/language/)
#
# SPDX-License-Identifier:    MIT

from __future__ import annotations

from typing import TYPE_CHECKING, final

from ..typing import ExpressionScalarType, NumericalScalarType
from .caching import cached
from .expressions import AbstractExpression, BindingExpression
from .indices import Index
from .scalar_types import common_scalar_type, numerical_type
from .structure import common_operand_domain

if TYPE_CHECKING:
    from ..typing.nodes import ArgumentSpaces
    from .domains import AbstractDomain
    from .indices import AbstractExtent, GatheredIndex
    from .topology import AbstractTopology


@final
class BroadcastTo(AbstractExpression):
    """Repeat a value along leading and singleton axes."""

    identifier = "uflx.BroadcastTo"
    value_fields = ("value",)

    value: AbstractExpression
    _value_shape: tuple[int | AbstractExtent, ...]

    @property
    def value_shape(self) -> tuple[int | AbstractExtent, ...]:
        return self._value_shape

    @property
    @cached
    def free_indices(self) -> tuple[tuple[Index, int | AbstractExtent], ...]:
        return self.value.free_indices

    @property
    @cached
    def argument_spaces(self) -> ArgumentSpaces:
        return self.value.argument_spaces

    @property
    @cached
    def scalar_type(self) -> ExpressionScalarType:
        return self.value.scalar_type

    @property
    @cached
    def domain(self) -> AbstractDomain | AbstractTopology | None:
        return self.value.domain

    def _validate(self) -> None:
        from .validation import extent

        for n in self.value_shape:
            extent(n)
        source = self.value.value_shape
        if len(source) > len(self.value_shape):
            raise ValueError("broadcast target has too few axes")
        aligned_axes = zip(reversed(source), reversed(self.value_shape))
        compatible = all(
            source_extent == target_extent or source_extent == 1
            for source_extent, target_extent in aligned_axes
        )
        if not compatible:
            raise ValueError("broadcast axes must match or have source extent one")


@final
class Tensor(AbstractExpression):
    requires_matching_domains = True
    value_fields = ("entries",)
    identifier = "uflx.Tensor"
    entries: tuple[AbstractExpression, ...]

    @property
    @cached
    def value_shape(self) -> tuple[int | AbstractExtent, ...]:
        return (len(self.entries), *self.entries[0].value_shape)

    @property
    @cached
    def free_indices(self) -> tuple[tuple[Index, int | AbstractExtent], ...]:
        from .static import free_indices

        return free_indices(*self.entries)

    @property
    @cached
    def argument_spaces(self) -> ArgumentSpaces:
        return self.entries[0].argument_spaces

    @property
    @cached
    def scalar_type(self) -> ExpressionScalarType:
        return common_scalar_type(*self.entries)

    @property
    def domain(self) -> AbstractDomain | AbstractTopology | None:
        return common_operand_domain(self)

    def _validate(self) -> None:
        if not self.entries:
            raise ValueError("tensor entries must be nonempty")
        if any(
            f.value_shape != self.entries[0].value_shape
            or f.argument_spaces != self.entries[0].argument_spaces
            for f in self.entries
        ):
            raise ValueError("tensor entry types differ")


@final
class Indexed(AbstractExpression):
    index_fields = ("indices",)
    value_fields = ("expression",)
    identifier = "uflx.Indexed"
    expression: AbstractExpression
    indices: tuple[int | Index | GatheredIndex, ...]

    @property
    @cached
    def free_indices(self) -> tuple[tuple[Index, int | AbstractExtent], ...]:
        from .validation import entries, merge_indices

        return merge_indices(
            self.expression.free_indices, entries(self.expression.value_shape, self.indices)
        )

    @property
    @cached
    def value_shape(self) -> tuple[int | AbstractExtent, ...]:
        return ()

    @property
    @cached
    def argument_spaces(self) -> ArgumentSpaces:
        return self.expression.argument_spaces

    @property
    @cached
    def scalar_type(self) -> ExpressionScalarType:
        return self.expression.scalar_type

    @property
    @cached
    def domain(self) -> AbstractDomain | AbstractTopology | None:
        return self.expression.domain

    def _validate(self) -> None:
        from .validation import entries, merge_indices

        if not self.indices:
            raise ValueError("Indexed requires nonempty entries")
        merge_indices(
            self.expression.free_indices, entries(self.expression.value_shape, self.indices)
        )


@final
class ComponentTensor(BindingExpression[Index]):
    value_fields = ("expression",)
    identifier = "uflx.ComponentTensor"
    expression: AbstractExpression
    indices: tuple[Index, ...]
    binding_fields = ("indices",)

    @property
    @cached
    def value_shape(self) -> tuple[int | AbstractExtent, ...]:
        free = dict(self.expression.free_indices)
        return tuple(free[i] for i in self.indices)

    @property
    @cached
    def free_indices(self) -> tuple[tuple[Index, int | AbstractExtent], ...]:
        return tuple((i, n) for i, n in self.expression.free_indices if i not in self.indices)

    @property
    @cached
    def argument_spaces(self) -> ArgumentSpaces:
        return self.expression.argument_spaces

    @property
    @cached
    def scalar_type(self) -> ExpressionScalarType:
        return self.expression.scalar_type

    @property
    @cached
    def domain(self) -> AbstractDomain | AbstractTopology | None:
        return self.expression.domain

    def _validate(self) -> None:
        _validate_index_binding(self)


@final
class IndexSum(BindingExpression[Index]):
    value_fields = ("expression",)
    identifier = "uflx.IndexSum"
    expression: AbstractExpression
    indices: tuple[Index, ...]
    binding_fields = ("indices",)

    @property
    @cached
    def free_indices(self) -> tuple[tuple[Index, int | AbstractExtent], ...]:
        return tuple((i, n) for i, n in self.expression.free_indices if i not in self.indices)

    @property
    @cached
    def value_shape(self) -> tuple[int | AbstractExtent, ...]:
        return ()

    @property
    @cached
    def argument_spaces(self) -> ArgumentSpaces:
        return self.expression.argument_spaces

    @property
    @cached
    def scalar_type(self) -> NumericalScalarType:
        return numerical_type(self.expression.scalar_type)

    @property
    @cached
    def domain(self) -> AbstractDomain | AbstractTopology | None:
        return self.expression.domain

    def _validate(self) -> None:
        _validate_index_binding(self)


@final
class Inverse(AbstractExpression):
    value_fields = ("matrix",)
    identifier = "uflx.Inverse"
    matrix: AbstractExpression

    @property
    @cached
    def value_shape(self) -> tuple[int | AbstractExtent, ...]:
        return tuple(reversed(self.matrix.value_shape))

    @property
    @cached
    def free_indices(self) -> tuple[tuple[Index, int | AbstractExtent], ...]:
        return self.matrix.free_indices

    @property
    @cached
    def argument_spaces(self) -> ArgumentSpaces:
        return ()

    @property
    @cached
    def scalar_type(self) -> NumericalScalarType:
        return numerical_type(self.matrix.scalar_type)

    @property
    @cached
    def domain(self) -> AbstractDomain | AbstractTopology | None:
        return self.matrix.domain

    def _validate(self) -> None:
        _matrix_operand(self.matrix, rectangular=True)


@final
class Determinant(AbstractExpression):
    value_fields = ("matrix",)
    identifier = "uflx.Determinant"
    matrix: AbstractExpression

    @property
    @cached
    def value_shape(self) -> tuple[int | AbstractExtent, ...]:
        return ()

    @property
    @cached
    def free_indices(self) -> tuple[tuple[Index, int | AbstractExtent], ...]:
        return self.matrix.free_indices

    @property
    @cached
    def argument_spaces(self) -> ArgumentSpaces:
        return ()

    @property
    @cached
    def scalar_type(self) -> NumericalScalarType:
        return numerical_type(self.matrix.scalar_type)

    @property
    @cached
    def domain(self) -> AbstractDomain | AbstractTopology | None:
        return self.matrix.domain

    def _validate(self) -> None:
        _matrix_operand(self.matrix)


def _validate_index_binding(e: ComponentTensor | IndexSum) -> None:
    if not e.indices or len(set(e.indices)) != len(e.indices):
        raise ValueError("bound indices must be nonempty and distinct")
    if e.expression.value_shape:
        raise ValueError("index binding requires a scalar")
    if any(i not in dict(e.expression.free_indices) for i in e.indices):
        raise ValueError("bound index is not free")


def _matrix_operand(f: AbstractExpression, rectangular: bool = False) -> None:
    if f.argument_spaces:
        raise ValueError("nonlinear operand has argument spaces")
    if len(f.value_shape) != 2:
        raise ValueError("matrix operand must have rank two")
    a, b = f.value_shape
    if type(a) is not int or type(b) is not int:
        raise ValueError("matrix dimensions must be fixed integers")
    if a != b and (not rectangular or b >= a or f.scalar_type != "real"):
        raise ValueError("matrix dimensions or scalar type are invalid")
