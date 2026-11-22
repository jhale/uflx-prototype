"""The boundary between interpreted operands and numerical Python callbacks."""

from collections.abc import Callable
from contextvars import ContextVar
from dataclasses import dataclass

import numpy as np

from . import _dual as ad

callback_points = ContextVar("uflx_oracle_callback_points", default={})


def call_at(points, fn, *args):
    token = callback_points.set(points)
    try:
        return fn(*args)
    finally:
        callback_points.reset(token)


def at(domain, point, fn):
    """Call fn with a point bound on domain."""
    return call_at(callback_points.get() | {domain: point}, fn)


@dataclass(frozen=True)
class Operand:
    """A delayed UFLx operand. Its body is evaluated by the interpreter."""

    read: Callable
    variations: Callable = frozenset

    def __call__(self, *args):
        return self.read(*args)


def tags(value):
    if isinstance(value, Operand):
        return value.variations()
    if isinstance(value, tuple):
        return set().union(*(tags(v) for v in value))
    return ad.tags(value)


def parts(value, tag):
    """Split a value or an entire field into its primal and tangent operands."""
    if isinstance(value, tuple):
        pairs = [parts(v, tag) for v in value]
        return tuple(p[0] for p in pairs), tuple(p[1] for p in pairs)
    if callable(value):

        def component(i):
            def read(*args):
                return parts(call(callback_points.get(), value, *args), tag)[i]

            return Operand(read, lambda: tags(value) - {tag})

        return component(0), component(1)
    return ad.split(value, tag)


def numerical(value):
    """A library may sample an operand or apply it to its own numerical arguments."""
    if isinstance(value, ad.Dual):
        if ad.is_zero(value.tangent):
            return numerical(value.value)
        raise TypeError(
            "use an ExternalOperator with a declared derivative action for this callback"
        )
    if isinstance(value, tuple):
        return tuple(numerical(v) for v in value)
    if callable(value):
        return lambda *args: numerical(value(*args))
    return value


def call(points, fn, *args):
    """Only interpreted operands accept internal derivative values."""
    if isinstance(fn, Operand):
        return call_at(points, fn, *args)
    locations = {domain: ad.primal(point) for domain, point in points.items()}
    return call_at(locations, fn, *(numerical(arg) for arg in args))


def sample(field, points, point):
    result = call(points, field, point)
    return np.asarray(result) if isinstance(result, (list, tuple)) else result
