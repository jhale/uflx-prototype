"""Lexical form slots, supplied fields and simultaneous directional derivatives."""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from uflx._src.algorithms.substitution import occurs_in_values
from uflx.nodes import AbstractExpression, Conjugate

from . import _dual as ad
from . import _finite_elements as elements
from ._context import call_at, callback_points


@dataclass(frozen=True)
class ExpressionBinding:
    """A replacement expression evaluated at its use site."""

    read: Callable


def variation_tags(operand, environment):
    """Find active variations of a whole field before a library samples it."""
    tags = set()
    for node, binding in environment.items():
        value = binding.read if isinstance(binding, ExpressionBinding) else binding
        if isinstance(node, AbstractExpression) and (active := ad.tags(value)):
            if occurs_in_values(node, operand):
                tags.update(active)
    return tags


def members(entry):
    return entry if isinstance(entry, tuple) else (entry,)


def conjugated(value):
    if isinstance(value, tuple):
        return tuple(conjugated(v) for v in value)
    if callable(value):

        def field(*args):
            return conjugated(value(*(conjugated(arg) for arg in args)))

        field._derivative_tags = lambda: ad.tags(value)
        return field
    return np.conj(value)


def function_value(node, frame, arguments):
    space = node.space
    if elements.is_basix_space(space):
        return elements.field_value(node, frame)
    value = frame.environment[node]
    if space.argument_spaces:
        value = call_at(frame.points, value, *arguments)
    if space.domain is not None:
        points = frame.points
        if isinstance(value, dict):
            value = value[points.get(("piece", space.domain), 0)]
        return (
            call_at(points, ad.call_field, value, points[space.domain])
            if callable(value)
            else ad.normalize(value)
        )
    return ad.normalize(
        call_at(frame.points, value) if callable(value) and not space.argument_spaces else value
    )


def supplied(operand, signature, frame):
    if isinstance(operand, tuple):
        return tuple(supplied(m, s, frame) for m, s in zip(operand, signature, strict=True))

    def apply(*slots):
        def read(local):
            value = local.value(operand, slots)
            return ad.asarray(value, dtype=complex) if signature.scalar_type == "complex" else value

        if signature.domain is not None:

            def field(point):
                return read(frame.at(callback_points.get() | {signature.domain: point}))

            field._derivative_tags = lambda: variation_tags(operand, frame.environment)
            return field
        return read(frame.at(callback_points.get()) if slots else frame)

    apply._derivative_tags = lambda: variation_tags(operand, frame.environment)
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
    tag = ad.fresh_tag()

    def seed(point, direction):
        def varied(caller, slots):
            local = frame.at(caller.points)
            return ad.pair(tag, local.value(point, slots), local.value(direction, slots))

        varied._derivative_tags = lambda: (
            {tag}
            | variation_tags(point, frame.environment)
            | variation_tags(direction, frame.environment)
        )
        return ExpressionBinding(varied)

    overrides = {
        variable: seed(p, d)
        for variable, p, d in zip(
            members(variables), members(point), members(direction), strict=True
        )
    }
    result = frame.bind_arguments(overrides).value(body, arguments)
    return ad.split(result, tag)[1]
