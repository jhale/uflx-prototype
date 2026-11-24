# Copyright (C) 2026 Jack S. Hale
#
# This file is part of uflx-core-experimental (https://fenics.github.io/uflx/language/)
#
# SPDX-License-Identifier:    MIT
"""Boolean comparison and conditional selection."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, final

from ..typing import ExpressionScalarType
from .caching import cached
from .expressions import AbstractExpression
from .scalar_types import common_scalar_type
from .structure import common_operand_domain

if TYPE_CHECKING:
    from ..typing.nodes import ArgumentSpaces
    from .domains import AbstractDomain
    from .indices import AbstractExtent, Index
    from .topology import AbstractTopology


@final
class LessThan(AbstractExpression):
    identifier = "uflx.LessThan"
    value_fields = ("first", "second")
    requires_matching_domains = True
    first: AbstractExpression
    second: AbstractExpression

    @property
    def scalar_type(self) -> Literal["boolean"]:
        return "boolean"

    @property
    def value_shape(self) -> tuple[()]:
        return ()

    @property
    def argument_spaces(self) -> tuple[()]:
        return ()

    @property
    def domain(self) -> AbstractDomain | AbstractTopology | None:
        return common_operand_domain(self)

    @property
    @cached
    def free_indices(self) -> tuple[tuple[Index, int | AbstractExtent], ...]:
        from .static import free_indices

        return free_indices(self.first, self.second)

    def _validate(self) -> None:
        for value in (self.first, self.second):
            if value.scalar_type != "real" or value.value_shape or value.argument_spaces:
                raise ValueError("comparison requires real scalar fields")


@final
class Conditional(AbstractExpression):
    identifier = "uflx.Conditional"
    value_fields = ("predicate", "true_value", "false_value")
    requires_matching_domains = True
    predicate: AbstractExpression
    true_value: AbstractExpression
    false_value: AbstractExpression

    @property
    @cached
    def scalar_type(self) -> ExpressionScalarType:
        return common_scalar_type(self.true_value, self.false_value)

    @property
    def value_shape(self) -> tuple[int | AbstractExtent, ...]:
        return self.true_value.value_shape

    @property
    def argument_spaces(self) -> ArgumentSpaces:
        return self.true_value.argument_spaces

    @property
    def domain(self) -> AbstractDomain | AbstractTopology | None:
        return common_operand_domain(self)

    @property
    @cached
    def free_indices(self) -> tuple[tuple[Index, int | AbstractExtent], ...]:
        from .static import free_indices

        return free_indices(self.predicate, self.true_value, self.false_value)

    def _validate(self) -> None:
        if (
            self.predicate.scalar_type != "boolean"
            or self.predicate.value_shape
            or self.predicate.argument_spaces
        ):
            raise ValueError("selection requires a scalar Boolean predicate")
        if (
            self.true_value.value_shape != self.false_value.value_shape
            or self.true_value.argument_spaces != self.false_value.argument_spaces
        ):
            raise ValueError("conditional branches must have equal shapes and argument spaces")
