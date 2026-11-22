"""Point bindings and memoization for a single evaluation."""

from contextvars import ContextVar

import numpy as np

from ._semantics import interpret

# The callback API can nest calls to evaluate(), including inside user-supplied fields.
_current_points = ContextVar("uflx_oracle_points", default=None)


def at(domain, point, fn):
    """Call fn with a point bound on domain, restoring the outer binding on return."""
    token = _current_points.set((_current_points.get() or {}) | {domain: point})
    try:
        return fn()
    finally:
        _current_points.reset(token)


class Evaluation:
    def __init__(self, interpreter, points, cells):
        self.spatial_step = interpreter.spatial_step
        self.derivative_step = interpreter.derivative_step
        self.points = dict(_current_points.get() or {})
        self.points.update((domain, np.asarray(point)) for domain, point in points.items())
        self.points.update((("piece", domain), slot) for domain, slot in cells.items())
        self.memo = {}
        self.location = ()

    def at(self, bindings, fn):
        points, location = self.points, self.location
        self.points = points | bindings
        self.location = tuple((domain, point_key(p)) for domain, p in self.points.items())
        token = _current_points.set(self.points)
        try:
            return fn()
        finally:
            self.points, self.location = points, location
            _current_points.reset(token)


def point_key(point):
    if isinstance(point, (tuple, list, np.ndarray)):
        array = np.asarray(point)
        return array.dtype, array.shape, array.tobytes()
    return point


class Frame:
    """Lexical bindings share the caller's current spatial point during evaluation."""

    def __init__(self, context, environment, indices, overrides, scope=None):
        self.context = context
        self.environment = environment
        self.indices = indices
        self.overrides = overrides
        self.scope = object() if scope is None else scope

    def value(self, node, arguments=()):
        if arguments:
            return self._read(node, arguments)
        indices = tuple((i, self.indices[i]) for i, _ in node.free_indices if i in self.indices)
        key = (self.scope, node, indices, self.context.location)
        memo = self.context.memo
        if key not in memo:
            memo[key] = self._read(node, ())
        return memo[key]

    def _read(self, node, arguments):
        if node in self.overrides:
            value = self.overrides[node]
            return value(*arguments) if callable(value) else value
        return interpret(node, self, arguments)

    def input(self, node, label):
        """Library data may be supplied by node or by its runtime label."""
        return self.environment[node] if node in self.environment else self.environment[label]

    def bind_indices(self, indices):
        return Frame(
            self.context, self.environment, self.indices | indices, self.overrides, self.scope
        )

    def bind_arguments(self, bindings):
        overrides = {k: v for k, v in self.overrides.items() if k not in bindings}
        return Frame(self.context, self.environment | bindings, self.indices, overrides)

    def perturb(self, overrides):
        return Frame(self.context, self.environment, self.indices, self.overrides | overrides)
