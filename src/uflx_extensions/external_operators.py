# Copyright (C) 2026 Jack S. Hale
#
# This file is part of uflx-core-experimental (https://fenics.github.io/uflx/language/)
#
# SPDX-License-Identifier:    MIT
"""Library-defined external operations and their directional derivative recipe."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from functools import cached_property
from typing import NamedTuple, final

from typing_extensions import override

from uflx.algorithms import free_indices
from uflx.nodes import (
    AbstractDomain,
    AbstractExpression,
    AbstractExtent,
    AbstractTopology,
    Base,
    Index,
)
from uflx.typing import ExpressionScalarType

__all__ = [
    "AbstractOperation",
    "ExternalOperator",
    "OutputType",
    "apply_external_derivatives",
    "external_operator",
    "operation",
]


class OutputType(NamedTuple):
    value_shape: tuple[int | AbstractExtent, ...]
    scalar_type: ExpressionScalarType
    domain: AbstractDomain | AbstractTopology | None


class AbstractOperation(Base, ABC):
    @abstractmethod
    def output_type(self, operands: tuple[AbstractExpression, ...]) -> OutputType:
        """Validate the operand signature and describe the result, or raise ValueError."""
        ...

    def derivative_action(self) -> AbstractOperation:
        """Identify the operation taking all point values followed by all directions."""
        raise TypeError(f"no derivative action for {self.identifier}")


@final
class _NamedOperation(AbstractOperation):
    identifier = "uflx.extensions.NamedOperation"
    name: str
    arity: int

    @override
    def _validate(self) -> None:
        if not isinstance(self.name, str):
            raise TypeError("expected a string operation name")
        if type(self.arity) is not int or self.arity < 0:
            raise ValueError("arity must be a nonnegative integer")

    @override
    def output_type(self, operands: tuple[AbstractExpression, ...]) -> OutputType:
        if len(operands) != self.arity:
            raise ValueError(f"{self.name} expects {self.arity} operands, got {len(operands)}")
        return self._output_rule(operands)

    @property
    def _output_rule(self) -> Callable[[tuple[AbstractExpression, ...]], OutputType]:
        return vars(self)["_rule"]

    def __reduce__(self):
        return operation, (self.name, self.arity, self._output_rule)


def operation(
    name: str, arity: int, output_type: Callable[[tuple[AbstractExpression, ...]], OutputType]
) -> AbstractOperation:
    """Declare an external operation. Its rule validates operands and returns their result type.

    The name and arity identify the symbolic operation. Reuse the same rule when
    declaring it again. Bind the numerical implementation separately when tracing.
    """
    if not callable(output_type):
        raise TypeError("expected an output type rule")
    op = _NamedOperation(name, arity)
    # Keep Python callables outside the symbolic construction arguments.
    if vars(op).setdefault("_rule", output_type) is not output_type:
        raise ValueError(f"{name} with arity {arity} already has a different output type rule")
    return op


@final
class ExternalOperator(AbstractExpression):
    identifier = "uflx.extensions.ExternalOperator"
    value_fields = ("operands",)
    operation: AbstractOperation
    operands: tuple[AbstractExpression, ...]

    @cached_property
    def output_type(self) -> OutputType:
        return self.operation.output_type(self.operands)

    @property
    def value_shape(self) -> tuple[int | AbstractExtent, ...]:
        return self.output_type.value_shape

    @property
    def scalar_type(self) -> ExpressionScalarType:
        return self.output_type.scalar_type

    @property
    def domain(self) -> AbstractDomain | AbstractTopology | None:
        return self.output_type.domain

    @property
    def argument_spaces(self) -> tuple[()]:
        return ()

    @property
    def free_indices(self) -> tuple[tuple[Index, int | AbstractExtent], ...]:
        return free_indices(*self.operands)

    @override
    def _validate(self) -> None:
        if isinstance(self.operation, AbstractExpression):
            raise ValueError("external operation metadata cannot be an expression")


def external_operator(
    operation: AbstractOperation, operands: tuple[AbstractExpression, ...]
) -> ExternalOperator:
    return ExternalOperator(operation, operands)


def _tangent(
    expression: AbstractExpression,
    at: Callable[[AbstractExpression], AbstractExpression],
    derivative: Callable[[AbstractExpression], AbstractExpression],
) -> AbstractExpression | None:
    if isinstance(expression, ExternalOperator):
        points = tuple(map(at, expression.operands))
        directions = tuple(map(derivative, expression.operands))
        return external_operator(expression.operation.derivative_action(), points + directions)
    return None


def apply_external_derivatives(f: AbstractExpression) -> AbstractExpression:
    """Apply EO01 together with the core directional derivative rules."""
    from uflx.passes import expand_derivatives_with

    return expand_derivatives_with(f, _tangent)
