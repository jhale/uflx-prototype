"""Lexical form slots, supplied fields and simultaneous directional derivatives."""

import numpy as np

from uflx._src.algorithms.substitution import occurs_in_values
from uflx.nodes import AbstractExpression, Conjugate

from . import _dual as ad
from . import _finite_elements as elements
from ._context import Operand, call, callback_points, parts, sample, tags


class ExpressionBinding(Operand):
    """A replacement expression evaluated with the frame and slots at its use site."""


def variation_tags(operand, environment):
    """Find active variations of a whole field before a library samples it."""
    active = set()
    for node, binding in environment.items():
        if isinstance(node, AbstractExpression) and occurs_in_values(node, operand):
            for value in binding.values() if isinstance(binding, dict) else (binding,):
                if isinstance(value, AbstractExpression):
                    active.update(variation_tags(value, environment))
                else:
                    active.update(tags(value))
    return active


def members(entry):
    return entry if isinstance(entry, tuple) else (entry,)


def conjugated(value):
    if isinstance(value, tuple):
        return tuple(conjugated(v) for v in value)
    if isinstance(value, AbstractExpression):
        return Conjugate(value)
    if callable(value):

        def field(*args):
            arguments = tuple(conjugated(arg) for arg in args)
            return conjugated(call(callback_points.get(), value, *arguments))

        return Operand(field, lambda: tags(value))
    return ad.linear(np.conj, value)


def multilinear(apply, slots):
    """Vary one form slot at a time, using the declared linearity in each slot."""
    active = tags(slots)
    if not active:
        return apply(slots)
    tag = max(active)
    values, directions = zip(*(parts(slot, tag) for slot in slots), strict=True)
    tangent = 0
    for i, direction in enumerate(directions):
        if tag in tags(slots[i]):
            tangent = ad.add(
                tangent, multilinear(apply, (*values[:i], direction, *values[i + 1 :]))
            )
    return ad.pair(tag, multilinear(apply, values), tangent)


def function_value(node, frame, arguments):
    space = node.space
    domain = space.domain
    if elements.is_basix_space(space):
        return elements.field_value(node, frame)
    binding = frame.environment[node]
    if isinstance(binding, dict):
        binding = binding[frame.points.get(("piece", domain), 0)]
    if isinstance(binding, AbstractExpression):
        return frame.value(binding, arguments)

    def apply(slots):
        value = binding
        if space.argument_spaces:
            value = call(frame.points, binding, *slots)
            if isinstance(value, dict):
                value = value[frame.points.get(("piece", domain), 0)]
        if callable(value) and domain is not None:
            return sample(value, frame.points, frame.points[domain])
        if callable(value) and not space.argument_spaces:
            value = call(frame.points, value)
        return np.asarray(value) if isinstance(value, (tuple, list)) else value

    return multilinear(apply, arguments) if space.argument_spaces else apply(())


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

            return Operand(field, lambda: variation_tags(operand, frame.environment))
        return read(frame.at(callback_points.get()) if slots else frame)

    return (
        Operand(apply, lambda: variation_tags(operand, frame.environment))
        if operand.argument_spaces
        else apply()
    )


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

        def variations():
            return (
                {tag}
                | variation_tags(point, frame.environment)
                | variation_tags(direction, frame.environment)
            )

        return ExpressionBinding(varied, variations)

    overrides = {
        variable: seed(p, d)
        for variable, p, d in zip(
            members(variables), members(point), members(direction), strict=True
        )
    }
    result = frame.bind_arguments(overrides).value(body, arguments)
    return ad.split(result, tag)[1]
