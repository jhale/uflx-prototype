from __future__ import annotations

from typing import TYPE_CHECKING, Literal, final

from ..interning import _cached
from ..nodes import AbstractExpression, Function
from ..passes.derivatives import _deriv, apply_derivatives
from ..traversal import _bottom_up
from ..tuples import _members

if TYPE_CHECKING:
    from ..nodes.domains import AbstractDomain
    from ..nodes.indices import AbstractExtent, Index
    from ..nodes.spaces import AbstractFunctionSpace
    from ..nodes.topology import AbstractTopology


# Shared implementation helpers; the package public API is declared in __init__.py.
__all__ = ["_apply_shape_derivatives", "_lower_with_shape_derivatives", "_shape_derivative"]


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

    def bound(self) -> tuple[Function, ...]:
        return _members(self.variable)

    @property
    @_cached
    def value_shape(self) -> tuple[int | AbstractExtent, ...]:
        return self.expression.value_shape

    @property
    @_cached
    def free_indices(self) -> tuple[tuple[Index, int | AbstractExtent], ...]:
        edges = (self.expression, *_members(self.direction))
        return tuple(dict(p for x in edges for p in x.free_indices).items())

    @property
    @_cached
    def argument_spaces(
        self,
    ) -> tuple[AbstractFunctionSpace | tuple[AbstractFunctionSpace, ...], ...]:
        return self.expression.argument_spaces

    @property
    @_cached
    def scalar_type(self) -> Literal["real", "complex"]:
        return self.expression.scalar_type

    @property
    @_cached
    def domain(self) -> AbstractDomain | AbstractTopology | None:
        return self.expression.domain


def _shape_derivative(
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
                return apply_derivatives(_deriv(g, x, x, h))
            case _:
                return e

    return _bottom_up(f, lower)


def _lower_with_shape_derivatives(f: AbstractExpression) -> AbstractExpression:
    """Expose geometry, differentiate it, then re-enter checked core lowering.

    Unlike core lower, this extension recipe permits ShapeDerivative nodes
    during the first spatial lowering stage. It does not weaken the final
    core-normal-form contract.
    """
    from ..passes.actions import apply_actions
    from ..passes.integrals import expand_integrals
    from ..passes.lowering import _lower_spatial, lower

    exposed = _lower_spatial(expand_integrals(apply_derivatives(apply_actions(f))))
    return lower(_apply_shape_derivatives(exposed))
