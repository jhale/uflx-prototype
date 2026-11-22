"""Lexical form slots, supplied fields and simultaneous directional derivatives."""

import jax
import jax.numpy as jnp

from uflx.nodes import AbstractExpression, Conjugate

from . import _finite_elements as elements
from ._context import Operand, call, sample


class ExpressionBinding(Operand):
    """A replacement expression evaluated at the caller's sampling location."""


def members(entry):
    return entry if isinstance(entry, tuple) else (entry,)


def conjugated(value):
    if isinstance(value, tuple):
        return tuple(conjugated(v) for v in value)
    if isinstance(value, AbstractExpression):
        return Conjugate(value)
    if callable(value):

        def field(site, *args):
            arguments = tuple(conjugated(arg) for arg in args)
            return conjugated(call(site, value, *arguments))

        return Operand(field)
    return jnp.conj(value)


def function_value(node, frame, arguments):
    space = node.space
    domain = space.domain
    if elements.is_basix_space(space):
        return elements.field_value(node, frame)
    binding = frame.environment[node]
    if isinstance(binding, dict):
        binding = binding[frame.site.slot(domain)]
    if isinstance(binding, AbstractExpression):
        return frame.value(binding, arguments)

    value = binding
    if space.argument_spaces:
        value = call(frame.site, binding, *arguments)
        if isinstance(value, dict):
            value = value[frame.site.slot(domain)]
    if callable(value) and domain is not None:
        return sample(value, frame.site, domain)
    if callable(value) and not space.argument_spaces:
        value = call(frame.site, value)
    return jnp.asarray(value) if isinstance(value, (tuple, list)) else value


def supplied(operand, signature, frame):
    if isinstance(operand, tuple):
        return tuple(supplied(m, s, frame) for m, s in zip(operand, signature, strict=True))

    def apply(site, *slots):
        def read(at):
            value = frame.value(operand, slots, at=at)
            return (
                jnp.asarray(value, dtype=complex) if signature.scalar_type == "complex" else value
            )

        if signature.domain is not None:

            def field(site, point):
                return read(site.at({signature.domain: point}))

            return Operand(field)
        return read(site)

    return Operand(apply) if operand.argument_spaces else apply(frame.site)


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
    return frame.bind(bindings).value(body, arguments[:count])


def directional(body, variables, point, direction, frame, arguments):
    def perturbed(epsilon):
        def seed(point, direction):
            def varied(site, slots):
                return frame.value(point, slots, at=site) + epsilon * frame.value(
                    direction, slots, at=site
                )

            return ExpressionBinding(varied)

        overrides = {
            variable: seed(p, d)
            for variable, p, d in zip(
                members(variables), members(point), members(direction), strict=True
            )
        }
        value = frame.bind(overrides).value(body, arguments)
        # An integer literal denotes a real constant in a differentiable expression.
        return jnp.asarray(value, dtype=jnp.result_type(value, 0.0))

    return jax.jvp(perturbed, (0.0,), (1.0,))[1]
