"""The boundary between interpreted operands and numerical Python callbacks."""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from . import _dual as ad


@dataclass(frozen=True)
class Operand:
    """A delayed UFLx operand. Its body is evaluated by the interpreter."""

    read: Callable
    variations: Callable = frozenset

    def __call__(self, frame, *args):
        return self.read(frame, *args)


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
            def read(frame, *args):
                return parts(call(frame, value, *args), tag)[i]

            return Operand(read, lambda: tags(value) - {tag})

        return component(0), component(1)
    return ad.split(value, tag)


def numerical(value, frame):
    """A library may sample an operand or apply it to its own numerical arguments."""
    if isinstance(value, ad.Dual):
        if ad.is_zero(value.tangent):
            return numerical(value.value, frame)
        raise TypeError(
            "use an ExternalOperator with a declared derivative action for this callback"
        )
    if isinstance(value, tuple):
        return tuple(numerical(v, frame) for v in value)
    if callable(value):
        return lambda *args: numerical(call(frame, value, *args), frame)
    return value


def call(frame, fn, *args):
    """Only interpreted operands accept internal derivative values."""
    if isinstance(fn, Operand):
        return fn(frame, *args)
    return fn(*(numerical(arg, frame) for arg in args))


def sample(field, frame, domain):
    result = call(frame, field, frame.points[domain])
    return np.asarray(result) if isinstance(result, (list, tuple)) else result
