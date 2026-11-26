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
from uflx.forms import apply, form, function, partial
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
from uflx.passes import apply_derivatives
from uflx.tensors import index_sum, indexed
from uflx.typing import ExpressionScalarType

__all__ = [
    "AbstractOperation",
    "ExternalOperator",
    "OutputType",
    "apply_external_derivatives",
    "external_adjoint",
    "external_operator",
    "jvp",
    "linearize",
    "operation",
    "vjp",
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
        return _JVPAction(self)

    def vjp_action(self, operand: int) -> AbstractOperation:
        """Return the operation taking primal operands and an output cotangent for this operand."""
        return _VJPAction(self, operand)


@final
class _JVPAction(AbstractOperation):
    identifier = "uflx.extensions.JVPAction"
    primal: AbstractOperation

    @override
    def output_type(self, operands: tuple[AbstractExpression, ...]) -> OutputType:
        n = len(operands) // 2
        points, directions = operands[:n], operands[n:]
        result = self.primal.output_type(points)
        for x, h in zip(points, directions, strict=True):
            if h.value_shape != x.value_shape or h.domain not in (None, x.domain):
                raise ValueError("incompatible external tangent direction")
        return result


@final
class _VJPAction(AbstractOperation):
    identifier = "uflx.extensions.VJPAction"
    primal: AbstractOperation
    operand: int

    @override
    def output_type(self, operands: tuple[AbstractExpression, ...]) -> OutputType:
        *points, seed = operands
        result = self.primal.output_type(tuple(points))
        if seed.value_shape != result.value_shape:
            raise ValueError("external VJP seed must match the output shape")
        if not 0 <= self.operand < len(points):
            raise ValueError("invalid external VJP operand")
        x = points[self.operand]
        domain = result.domain if seed.domain is None else seed.domain
        return OutputType(x.value_shape, x.scalar_type, domain if x.domain is None else x.domain)


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


def linearize(residual: Form, wrt: Function) -> Form:
    """Return B(h, v) = D R(u; v)[h] without expanding any external derivatives.

    B.arguments gives the perturbation and test slots. Fill either slot with
    uflx.forms.partial, then call apply_external_derivatives. Filling the test
    slot requests a VJP; filling the perturbation slot requests a JVP.
    """
    if len(residual.arguments) != 1 or not isinstance(residual.arguments[0], Function):
        raise ValueError("expected a residual with one test function")
    if residual.expression.argument_spaces != () or residual.value_shape != ():
        raise ValueError("expected a scalar residual with one test slot")
    residual = freshen(residual, (wrt,))
    test = residual.arguments[0]
    direction = function(wrt.space, ("uflx.extensions.linearize", residual, wrt))
    derivative = directional_derivative(residual, wrt, direction)
    # Keep the test value outside the derivative, including when it depends on wrt.
    return form(apply(derivative, (test,)), (direction, test))


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
            adjoint = external_adjoint(residual, wrt, h)
            if point != wrt:
                at_point = substitute_values(adjoint, {wrt: point})
                assert isinstance(at_point, Form)
                adjoint = at_point
            return partial(adjoint, {adjoint.arguments[1]: seed})
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
def jvp(expression: Form, wrt: Function, direction: AbstractExpression) -> Form: ...


@overload
def jvp(
    expression: AbstractExpression, wrt: Function, direction: AbstractExpression
) -> AbstractExpression: ...


def jvp(
    expression: AbstractExpression, wrt: Function, direction: AbstractExpression
) -> AbstractExpression:
    """Push direction through expression, preserving the open slots of an explicit form."""
    if isinstance(expression, Form):
        residual = freshen(expression, (wrt, direction))
        if (
            len(residual.arguments) == 1
            and isinstance(residual.arguments[0], Function)
            and residual.value_shape == ()
            and residual.expression.argument_spaces == ()
        ):
            bilinear = freshen(linearize(residual, wrt), (direction,))
            applied = partial(bilinear, {bilinear.arguments[0]: direction})
            result = apply_external_derivatives(applied)
            assert isinstance(result, Form)
            return result
        return form(jvp(residual.expression, wrt, direction), residual.arguments)
    return apply_external_derivatives(directional_derivative(expression, wrt, direction))


def _pair(left: AbstractExpression, right: AbstractExpression) -> AbstractExpression:
    if left.value_shape != right.value_shape:
        raise ValueError("cotangent and value shapes differ")
    if left.value_shape == ():
        return left * right
    axes = derived_indices("uflx.external_pair", (left, right), len(left.value_shape))
    return index_sum(indexed(left, axes) * indexed(right, axes), axes)


def external_adjoint(residual: Form, variable: Function, direction: Function) -> Form:
    """Build a real transpose Jacobian at fixed geometry using external VJPs.

    Slots are (direction, residual test), so partial application to the test
    slot gives the adjoint action. Both slots remain symbolic during the
    differentiation, even if a later argument depends on the variable. The reverse
    traversal supports integrals, sums, scalar products, full tensor inner
    products and pointwise external calls. Core-only branches use ordinary UFLx
    differentiation. Each active external operand requests its own VJP action.
    """
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

    @cache
    def active_external(node: AbstractExpression) -> bool:
        return occurs_in_values(variable, node) and (
            isinstance(node, ExternalOperator)
            or any(map(active_external, expression_operands(node)))
        )

    def reverse(node: AbstractExpression, cotangent: AbstractExpression) -> AbstractExpression:
        if not active_external(node):
            return _pair(
                cotangent, apply_derivatives(directional_derivative(node, variable, direction))
            )
        match node:
            case ExternalOperator(law, operands):
                contributions: list[AbstractExpression] = []
                for i, operand in enumerate(operands):
                    if occurs_in_values(variable, operand):
                        if operand.scalar_type != "real" or node.scalar_type != "real":
                            raise ValueError("external VJP requires real operands and results")
                        bar = external_operator(law.vjp_action(i), (*operands, cotangent))
                        if bar.scalar_type != "real" or bar.domain not in (
                            None,
                            operand.domain,
                            node.domain,
                            cotangent.domain,
                        ):
                            raise ValueError("invalid external cotangent type")
                        if bar.value_shape != operand.value_shape:
                            raise ValueError("external cotangent must match the operand shape")
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
                weight = (
                    _pair(cotangent, factor)
                    if value.value_shape == () and factor.value_shape != ()
                    else cotangent * factor
                )
                terms.append(reverse(value, weight))
        return reduce(add, terms)

    body = reverse(residual.expression, num(1))
    return form(body, (direction, test))


def vjp(residual: Form, wrt: Function, seed: AbstractExpression) -> Form:
    """Pull seed through a real one-slot residual, returning a form on wrt's space.

    Apply the result positionally to a perturbation. Its bound slot is generated
    from the operands so it cannot capture a field in the residual or the seed.
    """
    bilinear = freshen(linearize(residual, wrt), (seed,))
    applied = partial(bilinear, {bilinear.arguments[1]: seed})
    result = apply_external_derivatives(applied)
    assert isinstance(result, Form)
    return result
