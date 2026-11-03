from __future__ import annotations

from functools import cached_property
from typing import final

from ..nodes import AbstractExpression, Function
from ..passes.derivatives import apply_derivatives, deriv
from ..substitution import bottom_up
from ..tuples import members


@final
class ShapeDerivative(AbstractExpression):
    value_fields = ("expression", "direction")
    identifier = "uflx.shape.ShapeDerivative"
    # A derivative in a function that the coordinates do not depend on commutes with the node.
    commuting = ("expression",)
    # The derivative of the expression when the coordinate function x of its meshes moves in the
    # direction h, a node of the language extension for shape derivatives.
    expression: AbstractExpression
    variable: Function | tuple[Function, ...]
    direction: AbstractExpression | tuple[AbstractExpression, ...]
    scope = ("expression", "variable")

    def bound(self):
        return members(self.variable)

    @cached_property
    def value_shape(self):
        return self.expression.value_shape

    @cached_property
    def free_indices(self):
        edges = (self.expression, *members(self.direction))
        return tuple(dict(p for x in edges for p in x.free_indices).items())

    @cached_property
    def argument_spaces(self):
        return self.expression.argument_spaces

    @cached_property
    def scalar_type(self):
        return self.expression.scalar_type

    @cached_property
    def domain(self):
        return self.expression.domain


def shape_derivative(f, x, h):
    return ShapeDerivative(f, x, h)


def apply_shape_derivatives(f):
    # After the pull-backs are expanded, the coordinate function occurs in the expression itself,
    # in the coordinates, the Jacobians and the densities, so the derivative is a directional
    # derivative with the domains fixed.
    def lower(e):
        match e:
            case ShapeDerivative(g, x, h):
                return apply_derivatives(deriv(g, x, x, h))
        return e

    return bottom_up(f, lower)


def lower_with_shape_derivatives(f):
    """Expose geometry, differentiate it, then re-enter checked core lowering.

    Unlike core lower, this extension recipe permits ShapeDerivative nodes
    during the first spatial lowering stage. It does not weaken the final
    core-normal-form contract.
    """
    from ..passes.actions import apply_actions
    from ..passes.gradients import lower as lower_spatial
    from ..passes.integrals import expand_integrals
    from ..passes.lowering import lower

    exposed = lower_spatial(expand_integrals(apply_derivatives(apply_actions(f))))
    return lower(apply_shape_derivatives(exposed))
