# Copyright (C) 2026 Jack S. Hale
#
# This file is part of uflx-core-experimental (https://fenics.github.io/uflx/language/)
#
# SPDX-License-Identifier:    MIT
"""Library-defined external operations and their directional derivative recipe."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from functools import cache, cached_property, reduce
from operator import add
from typing import NamedTuple, final, overload

from typing_extensions import override

from uflx.algorithms import (
    bottom_up,
    derived_indices,
    expression_operands,
    free_indices,
    freshen,
    occurs,
    occurs_in_values,
    substitute_values,
)
from uflx.arithmetic import num
from uflx.calculus import directional_derivative
from uflx.forms import form, function, partial
from uflx.nodes import (
    AbstractDomain,
    AbstractExpression,
    AbstractExtent,
    AbstractTopology,
    Apply,
    Base,
    DirectionalDerivative,
    Form,
    Function,
    Index,
    Indexed,
    IndexSum,
    Integral,
    Product,
    Sum,
)
from uflx.tensors import index_sum, indexed
from uflx.typing import ExpressionScalarType

__all__ = [
    "AbstractOperation",
    "ExternalOperator",
    "OutputType",
    "Pullback",
    "PullbackComponent",
    "apply_external_derivatives",
    "external_operator",
    "jvp",
    "operation",
    "output_type",
    "vjp",
]


class OutputType(NamedTuple):
    value_shape: tuple[int | AbstractExtent, ...]
    scalar_type: ExpressionScalarType
    domain: AbstractDomain | AbstractTopology | None


def output_type(
    value_shape: tuple[int | AbstractExtent, ...],
    scalar_type: ExpressionScalarType,
    domain: AbstractDomain | AbstractTopology | None,
) -> OutputType:
    """Describe the value returned by an external operation."""
    return OutputType(value_shape, scalar_type, domain)


class AbstractOperation(Base, ABC):
    @abstractmethod
    def output_type(self, operands: tuple[AbstractExpression, ...]) -> OutputType:
        """Validate the operand signature and describe the result, or raise ValueError."""
        ...

    def jvp_operator(self) -> AbstractOperation:
        """Identify the operation taking all point values followed by all directions."""
        return _JVPAction(self)

    def vjp_operator(self) -> Pullback:
        """Identify the pullback returning one cotangent per primal operand, in order."""
        return Pullback(self)


@final
class _JVPAction(AbstractOperation):
    identifier = "uflx.extensions.JVPAction"
    primal: AbstractOperation

    @override
    def output_type(self, operands: tuple[AbstractExpression, ...]) -> OutputType:
        n = len(operands) // 2
        points, directions = operands[:n], operands[n:]
        typed_points = []
        for x, h in zip(points, directions, strict=True):
            if h.value_shape != x.value_shape or (
                x.domain is not None and h.domain not in (None, x.domain)
            ):
                raise ValueError("incompatible external tangent direction")
            # A constant point can stand for a field on the direction's domain.
            typed_points.append(h if x.domain is None and h.domain is not None else x)
        return self.primal.output_type(tuple(typed_points))


@final
class Pullback(Base):
    """Binding for a callable taking the primal values followed by the output seed."""

    identifier = "uflx.extensions.Pullback"
    primal: AbstractOperation

    def __getitem__(self, operand: int) -> PullbackComponent:
        return PullbackComponent(self, operand)


@final
class PullbackComponent(AbstractOperation):
    """Symbolic selection of one input cotangent from a pullback."""

    identifier = "uflx.extensions.PullbackComponent"
    pullback: Pullback
    operand: int

    @override
    def output_type(self, operands: tuple[AbstractExpression, ...]) -> OutputType:
        *points, seed = operands
        result = self.pullback.primal.output_type(tuple(points))
        if seed.value_shape != result.value_shape:
            raise ValueError("external VJP seed must match the output shape")
        if seed.scalar_type != result.scalar_type:
            raise ValueError("external VJP seed must match the output scalar type")
        if result.domain is not None and seed.domain not in (None, result.domain):
            raise ValueError("external VJP seed has an incompatible domain")
        if not 0 <= self.operand < len(points):
            raise ValueError("invalid external VJP operand")
        x = points[self.operand]
        domain = result.domain if seed.domain is None else seed.domain
        return output_type(x.value_shape, x.scalar_type, domain if x.domain is None else x.domain)


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
        return external_operator(expression.operation.jvp_operator(), points + directions)
    return None


def _select_pullback(node: AbstractExpression) -> AbstractExpression:
    match node:
        case Form(
            Apply(
                DirectionalDerivative(
                    Form() as residual,
                    Function() as wrt,
                    AbstractExpression() as point,
                    Function() as h,
                ),
                (AbstractExpression() as seed,),
            ),
            (slot,),
        ) if slot == h:
            if occurs(h, seed) or occurs(h, residual) or occurs(h, point):
                return node
            return _residual_vjp(freshen(residual, (point,)), wrt, point, h, seed)
        case _:
            return node


def apply_external_derivatives(f: AbstractExpression) -> AbstractExpression:
    """Expand seeded reverse actions before expanding the remaining forward derivatives.

    A direct full application or an explicitly expanded full Jacobian uses
    the forward rule. A test-seeded partial form retains its reverse action
    when enclosed in a subsequent Apply.
    """
    from uflx.passes import expand_derivatives_with

    return expand_derivatives_with(bottom_up(f, _select_pullback), _tangent)


@overload
def jvp(
    expression: Form, wrt: Function, point: AbstractExpression, direction: AbstractExpression
) -> Form: ...


@overload
def jvp(
    expression: AbstractExpression,
    wrt: Function,
    point: AbstractExpression,
    direction: AbstractExpression,
) -> AbstractExpression: ...


def jvp(
    expression: AbstractExpression,
    wrt: Function,
    point: AbstractExpression,
    direction: AbstractExpression,
) -> AbstractExpression:
    """Push direction through expression at point, preserving its open form slots."""
    if isinstance(expression, Form):
        residual = freshen(expression, (wrt, point, direction))
        return form(jvp(residual.expression, wrt, point, direction), residual.arguments)
    return apply_external_derivatives(directional_derivative(expression, wrt, point, direction))


def _pair(left: AbstractExpression, right: AbstractExpression) -> AbstractExpression:
    if left.value_shape != right.value_shape:
        raise ValueError("cotangent and value shapes differ")
    if left.value_shape == ():
        return left * right
    axes = derived_indices("uflx.external_pair", (left, right), len(left.value_shape))
    return index_sum(indexed(left, axes) * indexed(right, axes), axes)


def _residual_vjp(
    residual: Form,
    variable: Function,
    point: AbstractExpression,
    direction: Function,
    seed: AbstractExpression,
) -> Form:
    """Differentiate with the test slot open, then supply its value."""
    if len(residual.arguments) != 1 or not isinstance(residual.arguments[0], Function):
        raise ValueError("expected a residual with one test function")
    test = residual.arguments[0]
    if (residual.value_shape, residual.free_indices, residual.expression.argument_spaces) != (
        (),
        (),
        (),
    ):
        raise ValueError("expected a scalar residual with one test slot")
    if any(x.scalar_type != "real" for x in (residual, test, variable, direction)):
        raise ValueError("external adjoint action requires real values")
    if variable == test or direction.space != variable.space or direction == variable:
        raise ValueError("expected a separate coefficient and a direction in its space")
    if direction == test or occurs(direction, residual.expression):
        raise ValueError("direction already occurs in the residual")

    body = _pullback(residual.expression, variable, point, direction, num(1))
    adjoint = form(body, (direction, test))
    result = partial(adjoint, {test: seed})
    assert isinstance(result, Form)
    return result


def _pullback(
    expression: AbstractExpression,
    variable: Function,
    point: AbstractExpression,
    direction: Function,
    seed: AbstractExpression,
) -> AbstractExpression:
    """Pair the fixed seed with the variation, using VJPs at active external calls."""
    from uflx.passes import apply_derivatives

    def at(node: AbstractExpression) -> AbstractExpression:
        return node if point == variable else substitute_values(node, {variable: point})

    @cache
    def active_external(node: AbstractExpression) -> bool:
        return occurs_in_values(variable, node) and (
            isinstance(node, ExternalOperator)
            or any(map(active_external, expression_operands(node)))
        )

    def reverse(node: AbstractExpression, cotangent: AbstractExpression) -> AbstractExpression:
        if not active_external(node):
            return _pair(
                cotangent,
                apply_derivatives(directional_derivative(node, variable, point, direction)),
            )
        match node:
            case ExternalOperator(law, operands):
                pullback = law.vjp_operator()
                primals = tuple(map(at, operands))
                contributions: list[AbstractExpression] = []
                for i, operand in enumerate(operands):
                    if not occurs_in_values(variable, operand):
                        continue
                    if operand.scalar_type != "real" or node.scalar_type != "real":
                        raise ValueError("external VJP requires real operands and results")
                    bar = external_operator(pullback[i], (*primals, cotangent))
                    contributions.append(reverse(operand, bar))
                return reduce(add, contributions)
            case Integral(body, domain) if cotangent.domain is None:
                return Integral(reverse(body, cotangent), domain)
            case Sum(left, right):
                return reverse(left, cotangent) + reverse(right, cotangent)
            case Product(left, right):
                factors = (left, right)
            case IndexSum(Product(Indexed(left, a), Indexed(right, b)), axes) if (
                a == b == axes and not any(i in axes for i, _ in free_indices(left, right))
            ):
                factors = (left, right)
            case _:
                raise TypeError(f"unsupported external adjoint through {type(node).__name__}")

        terms: list[AbstractExpression] = []
        for value, factor in (factors, factors[::-1]):
            if occurs_in_values(variable, value):
                factor = at(factor)
                weight = (
                    _pair(cotangent, factor)
                    if value.value_shape == () and factor.value_shape != ()
                    else cotangent * factor
                )
                terms.append(reverse(value, weight))
        return reduce(add, terms)

    return reverse(expression, seed)


def vjp(
    residual: AbstractExpression, wrt: Function, point: AbstractExpression, seed: AbstractExpression
) -> Form:
    """Pull a fixed seed through a real expression or one-slot residual at point.

    For an expression F, the resulting form pairs seed with DF(point)[h] over
    all output components. For a residual, seed supplies the test-slot value.
    """
    if isinstance(residual, Form) and residual.argument_spaces != ():
        residual = freshen(residual, (wrt, point, seed))
        direction = function(wrt.space, ("uflx.extensions.vjp", residual, wrt, point, seed))
        adjoint = _residual_vjp(residual, wrt, point, direction, seed)
        result = apply_external_derivatives(adjoint)
        assert isinstance(result, Form)
        return result

    objective = residual.expression if isinstance(residual, Form) else residual
    if objective.free_indices or objective.argument_spaces:
        raise ValueError("expected an expression without free indices or argument slots")
    if (
        seed.value_shape != objective.value_shape
        or seed.argument_spaces != ()
        or seed.domain not in (None, objective.domain)
    ):
        raise ValueError(
            "expected a seed matching the expression shape and compatible with its domain"
        )
    if any(x.scalar_type != "real" for x in (objective, wrt, seed)):
        raise ValueError("external adjoint action requires real values")
    direction = function(wrt.space, ("uflx.extensions.vjp", objective, wrt, point, seed))
    return form(_pullback(objective, wrt, point, direction, seed), (direction,))
