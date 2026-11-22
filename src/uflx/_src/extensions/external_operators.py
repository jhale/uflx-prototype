"""Library-defined external operations and their directional derivative recipe."""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, NamedTuple, final

from ..nodes import AbstractExpression, Base
from ..nodes.caching import cached
from ..nodes.scalar_types import ExpressionScalarType
from ..nodes.static import free_indices

if TYPE_CHECKING:
    from ..nodes.domains import AbstractDomain
    from ..nodes.indices import AbstractExtent, Index
    from ..nodes.topology import AbstractTopology
    from ..rules.differentiation import Transform


class OutputType(NamedTuple):
    value_shape: tuple[int | AbstractExtent, ...]
    scalar_type: ExpressionScalarType
    domain: AbstractDomain | AbstractTopology | None


class AbstractOperation(Base):
    @abstractmethod
    def output_type(self, operands: tuple[AbstractExpression, ...]) -> OutputType:
        """Validate the operand signature and describe the result, or raise ValueError."""
        ...

    def derivative_action(self) -> AbstractOperation:
        """Identify the operation taking all point values followed by all directions."""
        raise TypeError(f"no derivative action for {self.identifier}")


@final
class ExternalOperator(AbstractExpression):
    identifier = "uflx.extensions.ExternalOperator"
    value_fields = ("operands",)
    operation: AbstractOperation
    operands: tuple[AbstractExpression, ...]

    @property
    @cached
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
    @cached
    def free_indices(self) -> tuple[tuple[Index, int | AbstractExtent], ...]:
        return free_indices(*self.operands)

    def _validate(self) -> None:
        if isinstance(self.operation, AbstractExpression):
            raise ValueError("external operation metadata cannot be an expression")


def external_operator(
    operation: AbstractOperation, operands: tuple[AbstractExpression, ...]
) -> ExternalOperator:
    return ExternalOperator(operation, operands)


def _tangent(
    expression: AbstractExpression, at: Transform, derivative: Transform
) -> AbstractExpression | None:
    if isinstance(expression, ExternalOperator):
        points = tuple(map(at, expression.operands))
        directions = tuple(map(derivative, expression.operands))
        return external_operator(expression.operation.derivative_action(), points + directions)
    return None


def apply_external_derivatives(f: AbstractExpression) -> AbstractExpression:
    """Apply EO01 together with the core directional derivative rules."""
    from ..passes.derivatives import expand_derivatives_with

    return expand_derivatives_with(f, _tangent)
