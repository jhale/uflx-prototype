"""Evaluate UFLx expressions and fully applied forms by their numerical definitions."""

from collections.abc import Mapping

import jax
import jax.numpy as jnp

from uflx.nodes import AbstractDomain, AbstractExpression, AbstractTopology

from ._context import Sample
from ._external import Pointwise as Pointwise
from ._semantics import evaluate


def trace(expression: AbstractExpression, *inputs):
    """Return a JAX-traceable function for an expression or form.

    Call it with one value per declared input, then one per ordered form slot.
    A field input takes its coefficients, a domain input takes an evaluation point,
    and a free index input takes its selected component. The result is a JAX array.

    An external operation input takes its JAX-traceable implementation. Bind fixed
    inputs with functools.partial. To sample a particular cell, pull the expression
    back to its reference cell before tracing.
    """
    if len(set(inputs)) != len(inputs):
        raise ValueError("each input must be declared once")
    arity = len(inputs) + len(expression.argument_spaces)

    def kernel(*values):
        if len(values) != arity:
            raise TypeError(f"expected {arity} positional values, got {len(values)}")
        environment = {}
        points = {}
        for key, value in zip(inputs, values[: len(inputs)], strict=True):
            if isinstance(key, (AbstractDomain, AbstractTopology)):
                points[key] = jnp.asarray(value, dtype=float)
            else:
                environment[key] = value
        sample = Sample(points, {})
        return jnp.asarray(evaluate(expression, environment, sample, values[len(inputs) :]))

    return kernel


class Interpreter:
    """Evaluate UFLx with JAX using the supplied field and operation bindings.

    Bind fields to numbers, arrays, UFLx expressions or JAX-traceable functions of a point.
    Blocked Basix fields accept flat coefficients or one block per scalar basis DOF,
    with trailing axes matching the space's block shape.
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
