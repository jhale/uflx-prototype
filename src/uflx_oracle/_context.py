"""Spatial bindings for Python callbacks that re-enter the interpreter."""

from contextvars import ContextVar

# Python field callbacks accept coordinates, but may also call the interpreter
# or sample another field. Make their caller's domain and cell bindings available.
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
