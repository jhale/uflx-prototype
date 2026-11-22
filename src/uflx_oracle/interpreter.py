"""Evaluate UFLx expressions and fully applied forms by their numerical definitions."""

from collections.abc import Mapping

from uflx.nodes import AbstractExpression

from . import _dual as ad
from ._external import Pointwise as Pointwise
from ._semantics import Frame


class Interpreter:
    """A numerical reference interpreter with field and operation bindings.

    Bind fields to numbers, arrays, UFLx expressions or numerical point callbacks.
    Use an ExternalOperator with a derivative action for a differentiable library function.
    Bind operator-valued functions to callables of their slots. Runtime arrays,
    coefficients and extents also accept their library labels.
    """

    def __init__(self, environment: Mapping | None = None):
        self.environment = environment if environment is not None else {}

    def evaluate(
        self, expression: AbstractExpression, arguments=(), *, indices=None, points=None, cells=None
    ):
        """Return the numerical value for the given ordered form slots.

        Supply free index values in indices, coordinates keyed by domain in points,
        and local cell slots keyed by mesh or topology in cells.
        """
        arguments = tuple(arguments)
        if len(arguments) != len(expression.argument_spaces):
            raise ValueError(
                f"expected {len(expression.argument_spaces)} form arguments, got {len(arguments)}"
            )
        frame = Frame(
            dict(self.environment),
            dict(indices or {}),
            {domain: ad.asarray(point) for domain, point in (points or {}).items()},
            dict(cells or {}),
        )
        return frame.value(expression, arguments)
