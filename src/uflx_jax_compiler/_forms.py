"""Lexical form slots, supplied fields and simultaneous directional derivatives."""

import jax
import jax.numpy as jnp

from uflx.nodes import AbstractExpression, Conjugate

from . import _finite_elements as elements
from ._context import BoundExpression, as_argument


def members(entry):
    return entry if isinstance(entry, tuple) else (entry,)


def conjugated(value):
    if isinstance(value, tuple):
        return tuple(conjugated(v) for v in value)
    if isinstance(value, AbstractExpression):
        return Conjugate(value)
    if isinstance(value, BoundExpression):

        def read(site, slots):
            return conjugated(value.read(site, tuple(conjugated(arg) for arg in slots)))

        return BoundExpression(value.signature, read)
    if callable(value):
        return lambda *args: conjugated(value(*(conjugated(arg) for arg in args)))
    return jnp.conj(value)


def function_value(node, bindings, site, arguments, read):
    space = node.space
    domain = space.domain
    if elements.is_basix_space(space):
        return elements.field_value(node, bindings, site, read)
    value = bindings[node]
    if isinstance(value, dict):
        value = value[site.slot(domain)]
    if isinstance(value, AbstractExpression):
        return read(value, arguments)
    if space.argument_spaces:
        value = value(*(as_argument(arg, site) for arg in arguments))
        if isinstance(value, dict):
            value = value[site.slot(domain)]
    if callable(value) and domain is not None:
        value = value(site.points[domain])
    elif callable(value) and not space.argument_spaces:
        value = value()
    return jnp.asarray(value) if isinstance(value, (tuple, list)) else value


def supplied(operand, signature, site, value):
    if isinstance(operand, tuple):
        return tuple(supplied(m, s, site, value) for m, s in zip(operand, signature, strict=True))

    def read(at, slots):
        result = value(operand, slots, at=at)
        return jnp.asarray(result, dtype=complex) if signature.scalar_type == "complex" else result

    if signature.domain is None and not operand.argument_spaces:
        return read(site, ())
    return BoundExpression(signature, read)


def form_bindings(declarations, arguments):
    bindings = {}
    for entry, value in zip(declarations, arguments, strict=True):
        values = value if isinstance(entry, tuple) else (value,)
        for member, item in zip(members(entry), values, strict=True):
            if isinstance(member, Conjugate):
                bindings[member.operand] = conjugated(item)
            else:
                bindings[member] = item
    return bindings


def directional(node, arguments, read):
    def perturbed(epsilon):
        def seed(variable, point, direction):
            def varied(site, slots):
                return read(point, slots, at=site) + epsilon * read(direction, slots, at=site)

            return BoundExpression(variable, varied)

        overrides = {
            variable: seed(variable, p, d)
            for variable, p, d in zip(
                members(node.variable), members(node.point), members(node.direction), strict=True
            )
        }
        value = read(node.expression, arguments, bind=overrides)
        # An integer literal denotes a real constant in a differentiable expression.
        return jnp.asarray(value, dtype=jnp.result_type(value, 0.0))

    return jax.jvp(perturbed, (0.0,), (1.0,))[1]
