"""Lexical form slots, supplied fields and simultaneous directional derivatives."""

import numpy as np

from uflx.nodes import Conjugate

from . import _finite_elements as elements


def members(entry):
    return entry if isinstance(entry, tuple) else (entry,)


def conjugated(value):
    if isinstance(value, tuple):
        return tuple(conjugated(v) for v in value)
    if callable(value):
        return lambda *args: conjugated(value(*(conjugated(arg) for arg in args)))
    return np.conj(value)


def function_value(node, frame, arguments):
    space = node.space
    if elements.is_basix_space(space):
        return elements.field_value(node, frame)
    value = frame.environment[node]
    if space.argument_spaces:
        value = value(*arguments)
    if space.domain is not None:
        points = frame.context.points
        if isinstance(value, dict):
            value = value[points.get(("piece", space.domain), 0)]
        return value(points[space.domain]) if callable(value) else value
    return value() if callable(value) and not space.argument_spaces else value


def supplied(operand, signature, frame):
    if isinstance(operand, tuple):
        return tuple(supplied(m, s, frame) for m, s in zip(operand, signature, strict=True))

    def apply(*slots):
        def read():
            value = frame.value(operand, slots)
            return np.asarray(value, dtype=complex) if signature.scalar_type == "complex" else value

        if signature.domain is not None:
            return lambda point: frame.context.at({signature.domain: point}, read)
        return read()

    return apply if operand.argument_spaces else apply()


def form_value(body, declarations, frame, arguments):
    count = len(body.argument_spaces)
    bindings = {}
    for entry, value in zip(declarations, arguments[count:], strict=True):
        values = value if isinstance(entry, tuple) else (value,)
        for member, item in zip(members(entry), values, strict=True):
            if isinstance(member, Conjugate):
                bindings[member.operand] = conjugated(item)
            else:
                bindings[member] = item
    return frame.bind_arguments(bindings).value(body, arguments[:count])


def directional(body, variables, point, direction, frame, arguments):
    step = frame.context.derivative_step

    def displaced(delta):
        overrides = {
            variable: (
                lambda *slots, p=p, d=d: frame.value(p, slots) + delta * frame.value(d, slots)
            )
            for variable, p, d in zip(
                members(variables), members(point), members(direction), strict=True
            )
        }
        return frame.perturb(overrides).value(body, arguments)

    return (displaced(step) - displaced(-step)) / (2 * step)
