# Copyright (C) 2026 Jack S. Hale
#
# This file is part of uflx-core-experimental (https://fenics.github.io/uflx/language/)
#
# SPDX-License-Identifier:    MIT
"""Boolean comparison and conditional selection."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, final

from .caching import cached
from .expressions import AbstractExpression
from .scalar_types import ExpressionScalarType
from .validation import require

if TYPE_CHECKING:
    from .domains import AbstractDomain
    from .indices import AbstractExtent, Index
    from .spaces import AbstractFunctionSpace
    from .topology import AbstractTopology


@final
class LessThan(AbstractExpression):
    identifier = "uflx.LessThan"
    value_fields = ("first", "second")
    matching_domains = True
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
    @cached
    def domain(self) -> AbstractDomain | AbstractTopology | None:
        return self.first.domain or self.second.domain

    @property
    @cached
    def free_indices(self) -> tuple[tuple[Index, int | AbstractExtent], ...]:
        from .static import free_indices

        return free_indices(self.first, self.second)

    def _validate(self) -> None:
        for value in (self.first, self.second):
            require(
                value.scalar_type == "real" and not value.value_shape and not value.argument_spaces,
                "comparison requires real scalar fields",
            )


@final
class Conditional(AbstractExpression):
    identifier = "uflx.Conditional"
    value_fields = ("predicate", "true_value", "false_value")
    matching_domains = True
    predicate: AbstractExpression
    true_value: AbstractExpression
    false_value: AbstractExpression

    @property
    @cached
    def scalar_type(self) -> ExpressionScalarType:
        return self.joined_scalar_type(self.true_value, self.false_value)

    @property
    def value_shape(self) -> tuple[int | AbstractExtent, ...]:
        return self.true_value.value_shape

    @property
    def argument_spaces(
        self,
    ) -> tuple[AbstractFunctionSpace | tuple[AbstractFunctionSpace, ...], ...]:
        return self.true_value.argument_spaces

    @property
    @cached
    def domain(self) -> AbstractDomain | AbstractTopology | None:
        return self.predicate.domain or self.true_value.domain or self.false_value.domain

    @property
    @cached
    def free_indices(self) -> tuple[tuple[Index, int | AbstractExtent], ...]:
        from .static import free_indices

        return free_indices(self.predicate, self.true_value, self.false_value)

    def _validate(self) -> None:
        require(
            self.predicate.scalar_type == "boolean"
            and not self.predicate.value_shape
            and not self.predicate.argument_spaces,
            "selection requires a scalar Boolean predicate",
        )
        require(
            self.true_value.value_shape == self.false_value.value_shape
            and self.true_value.argument_spaces == self.false_value.argument_spaces,
            "conditional branches must have equal shapes and argument spaces",
        )
