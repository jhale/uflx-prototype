"""Evaluate UFLx expressions and fully applied forms by their numerical definitions."""

from collections.abc import Mapping

import jax
import jax.numpy as jnp

from uflx.nodes import AbstractExpression

from ._context import Sample
from ._external import Pointwise as Pointwise
from ._semantics import evaluate


class Interpreter:
    """A numerical reference interpreter with field and operation bindings.

    Bind fields to numbers, arrays, UFLx expressions or JAX-traceable functions of a point.
    Bind external operations to JAX-traceable functions. A field operand is a callable
    of its coordinates. Wrap a function in Pointwise to pass values at the current point.
    Bind operator-valued functions to callables of their slots. Domain-free input terminals
    with no free indices or slots accept arrays. Inputs and extents also accept library labels.
    """

    def __init__(self, environment: Mapping | None = None):
        self.environment = environment if environment is not None else {}

    def evaluate(
        self, expression: AbstractExpression, arguments=(), *, indices=None, points=None, cells=None
    ):
        """Return the numerical value for the given ordered form slots.

        Unselected free indices are leading array axes in expression.free_indices order.
        Select an entry with indices={i: k}. Supply coordinates keyed by domain in points.
        When a point evaluation needs one of several local cells, supply its slot
        in cells, keyed by mesh or topology.
        """
        arguments = tuple(arguments)
        if len(arguments) != len(expression.argument_spaces):
            raise ValueError(
                f"expected {len(expression.argument_spaces)} form arguments, got {len(arguments)}"
            )
        site = Sample(
            {domain: jnp.asarray(point, dtype=float) for domain, point in (points or {}).items()},
            dict(cells or {}),
        )
        bindings = dict(self.environment) | dict(indices or {})
        value = evaluate(expression, bindings, site, arguments)
        # Eager scalar results can be used as UFLx literals. A traced scalar stays in the graph.
        if (
            isinstance(value, jax.Array)
            and not isinstance(value, jax.core.Tracer)
            and not value.ndim
        ):
            return value.item()
        return value
