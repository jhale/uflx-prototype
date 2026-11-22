"""Evaluate UFLx expressions and fully applied forms by their numerical definitions."""

from collections.abc import Mapping
from math import isfinite

from uflx.nodes import AbstractExpression

from ._context import Evaluation, Frame
from ._context import at as at


class Interpreter:
    """A reference interpreter with field bindings and central-difference step sizes.

    Bind functions to numbers, arrays or callables of a point. Bind operator-valued
    functions to callables of their slots. Runtime arrays, coefficients and extents
    also accept their library labels. Each call reads the current environment.
    """

    def __init__(
        self,
        environment: Mapping | None = None,
        *,
        spatial_step: float = 1e-5,
        derivative_step: float = 1e-4,
    ):
        if not isfinite(spatial_step) or spatial_step <= 0:
            raise ValueError("spatial_step must be finite and positive")
        if not isfinite(derivative_step) or derivative_step <= 0:
            raise ValueError("derivative_step must be finite and positive")
        self.environment = environment if environment is not None else {}
        self.spatial_step = spatial_step
        self.derivative_step = derivative_step

    def evaluate(
        self, expression: AbstractExpression, arguments=(), *, indices=None, points=None, cells=None
    ):
        """Return the numerical value for the given ordered form slots.

        Supply free index values in indices, coordinates keyed by domain in points,
        and local cell slots keyed by mesh or topology in cells.
        """
        return self._evaluate(expression, arguments, indices, points, cells, {})

    def _evaluate(self, expression, arguments, indices, points, cells, overrides):
        arguments = tuple(arguments)
        if len(arguments) != len(expression.argument_spaces):
            raise ValueError(
                f"expected {len(expression.argument_spaces)} form arguments, got {len(arguments)}"
            )
        context = Evaluation(self, points or {}, cells or {})
        frame = Frame(context, dict(self.environment), dict(indices or {}), overrides)
        try:
            return context.at({}, lambda: frame.value(expression, arguments))
        finally:
            context.memo.clear()


def evaluate(e, env, ienv=None, over=None):
    """Return a callable of the ordered form slots, evaluated afresh on each call."""
    interpreter = Interpreter(env)
    return lambda vals: interpreter._evaluate(e, vals, ienv, None, None, over or {})
