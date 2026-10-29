from __future__ import annotations

from functools import cached_property
from typing import final

from ..nodes import AbstractExpression, Function
from ..passes.commuting import COMMUTING
from ..passes.derivatives import deriv
from ..substitution import bottom_up
from ..tuples import members


@final
class ShapeDerivative(AbstractExpression):
    identifier = "uflx.shape.ShapeDerivative"
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
    def support(self):
        return self.expression.support


# A derivative in a function that the coordinates do not depend on commutes with the node.
COMMUTING[ShapeDerivative] = ("expression",)


def shape_derivative(f, x, h):
    return ShapeDerivative(f, x, h)


def apply_shape_derivatives(f):
    # After the pull-backs are expanded, the coordinate function occurs in the expression itself,
    # in the coordinates, the Jacobians and the densities, so the derivative is a directional
    # derivative with the domains fixed.
    def lower(e):
        match e:
            case ShapeDerivative(g, x, h):
                return deriv(g, x, x, h)
        return e

    return bottom_up(f, lower)
