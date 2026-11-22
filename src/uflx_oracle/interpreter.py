"""Evaluate UFLx expressions and fully applied forms by their numerical definitions."""

from collections.abc import Mapping

from uflx.nodes import AbstractExpression

from . import _dual as ad
from ._context import at as at
from ._context import call, callback_points
from ._external import Pointwise as Pointwise
from ._forms import ExpressionBinding
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
        locations = callback_points.get() | {
            domain: ad.asarray(point) for domain, point in (points or {}).items()
        }
        locations.update((("piece", domain), slot) for domain, slot in (cells or {}).items())
        return Frame(dict(self.environment), dict(indices or {}), locations).value(
            expression, arguments
        )


def evaluate(e, env, ienv=None, over=None):
    """Return a callable of the ordered form slots, evaluated afresh on each call."""

    def replacement(value):
        def read(frame, arguments):
            return call(frame.points, value, *arguments) if callable(value) else value

        return ExpressionBinding(read)

    def run(arguments):
        bindings = {node: replacement(value) for node, value in (over or {}).items()}
        return Interpreter(dict(env) | bindings).evaluate(e, arguments, indices=ienv)

    return run
