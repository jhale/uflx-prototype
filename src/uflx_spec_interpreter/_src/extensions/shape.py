from __future__ import annotations

from typing import TYPE_CHECKING, Literal, final

from ..algorithms.traversal import bottom_up
from ..nodes import AbstractExpression, Function
from ..nodes.arguments import argument_members
from ..nodes.caching import cached
from ..passes.derivatives import apply_derivatives
from ..rules.differentiation import deriv

if TYPE_CHECKING:
    from ..nodes.domains import AbstractDomain
    from ..nodes.indices import AbstractExtent, Index
    from ..nodes.spaces import AbstractFunctionSpace
    from ..nodes.topology import AbstractTopology


@final
class _ShapeDerivative(AbstractExpression):
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
    binding_fields = ("variable",)

    def bound(self) -> tuple[Function, ...]:
        return argument_members(self.variable)

    @property
    @cached
    def value_shape(self) -> tuple[int | AbstractExtent, ...]:
        return self.expression.value_shape

    @property
    @cached
    def free_indices(self) -> tuple[tuple[Index, int | AbstractExtent], ...]:
        edges = (self.expression, *argument_members(self.direction))
        from ..nodes.static import free_indices

        return free_indices(*edges)

    @property
    @cached
    def argument_spaces(
        self,
    ) -> tuple[AbstractFunctionSpace | tuple[AbstractFunctionSpace, ...], ...]:
        return self.expression.argument_spaces

    @property
    @cached
    def scalar_type(self) -> Literal["real", "complex"]:
        return self.expression.scalar_type

    @property
    @cached
    def domain(self) -> AbstractDomain | AbstractTopology | None:
        return self.expression.domain


def shape_derivative(
    f: AbstractExpression,
    x: Function | tuple[Function, ...],
    h: AbstractExpression | tuple[AbstractExpression, ...],
) -> _ShapeDerivative:
    return _ShapeDerivative(f, x, h)


def _apply_shape_derivatives(f: AbstractExpression) -> AbstractExpression:
    # After the pull-backs are expanded, the coordinate function occurs in the expression itself,
    # in the coordinates, the Jacobians and the densities, so the derivative is a directional
    # derivative with the domains fixed.
    def lower(e: AbstractExpression) -> AbstractExpression:
        match e:
            case _ShapeDerivative(g, x, h):
                return apply_derivatives(deriv(g, x, x, h))
            case _:
                return e

    return bottom_up(f, lower)


def lower_with_shape_derivatives(f: AbstractExpression) -> AbstractExpression:
    """Expose geometry, differentiate it, then re-enter checked core lowering.

    Unlike core lower, this extension recipe permits ShapeDerivative nodes
    during the first spatial lowering stage. It does not weaken the final
    core-normal-form contract.
    """
    from ..passes.actions import apply_actions
    from ..passes.integrals import expand_integrals
    from ..passes.lowering import lower, lower_spatial

    exposed = apply_actions(f)
    exposed = apply_derivatives(exposed)
    exposed = expand_integrals(exposed)
    exposed = lower_spatial(exposed)
    return lower(_apply_shape_derivatives(exposed))
