# Copyright (C) 2026 Jack S. Hale
#
# This file is part of uflx-core-experimental (https://fenics.github.io/uflx/language/)
#
# SPDX-License-Identifier:    MIT
"""Lexical form slots, supplied fields and simultaneous directional derivatives."""

import jax.numpy as jnp

from uflx.nodes import AbstractExpression, Conjugate

from . import _finite_elements as elements
from ._context import BoundExpression, as_argument


def conjugated(value):
    if isinstance(value, tuple):
        return tuple(conjugated(v) for v in value)
    if isinstance(value, AbstractExpression):
        return Conjugate(value)
    if isinstance(value, BoundExpression):

        def read(site, slots):
            return conjugated(value.read(site, tuple(conjugated(arg) for arg in slots)))

        return BoundExpression(
            value.signature, read, value.needs_eager_samples, value.is_basis_field
        )
    if callable(value):
        return lambda *args: conjugated(value(*(conjugated(arg) for arg in args)))
    return jnp.conj(value)


def function_value(node, bindings, site, arguments, read):
    space = node.space
    domain = space.domain
    element_space = elements.basix_space(space)
    if element_space is not None:
        return elements.field_value(node, element_space, bindings, site, read)
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

    def bind(entry, value):
        if isinstance(entry, tuple):
            for member, item in zip(entry, value, strict=True):
                bind(member, item)
        elif isinstance(entry, Conjugate):
            bindings[entry.operand] = conjugated(value)
        else:
            bindings[entry] = value

    for entry, value in zip(declarations, arguments, strict=True):
        bind(entry, value)
    return bindings
