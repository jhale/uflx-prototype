"""The boundary between interpreted operands and numerical Python callbacks."""

from collections.abc import Callable
from dataclasses import dataclass

import jax.numpy as jnp


@dataclass(frozen=True)
class Operand:
    """A delayed UFLx operand. Its body is evaluated by the interpreter."""

    read: Callable

    def __call__(self, frame, *args):
        return self.read(frame, *args)


def bound(value, frame):
    if isinstance(value, tuple):
        return tuple(bound(v, frame) for v in value)
    if isinstance(value, Operand):
        return lambda *args: bound(value(frame, *args), frame)
    return value


def call(frame, fn, *args):
    if isinstance(fn, Operand):
        return fn(frame, *args)
    return fn(*(bound(arg, frame) for arg in args))


def sample(field, frame, domain):
    result = call(frame, field, frame.points[domain])
    return jnp.asarray(result) if isinstance(result, (list, tuple)) else result
