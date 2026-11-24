# Copyright (C) 2026 Jack S. Hale
#
# This file is part of uflx-core-experimental (https://fenics.github.io/uflx/language/)
#
# SPDX-License-Identifier:    MIT

from __future__ import annotations

from typing import TYPE_CHECKING

from ..algorithms.rewriting import rewrite, zero_like
from ..algorithms.traversal import everywhere
from ..arithmetic import multiply
from ..forms import apply, partial
from ..nodes import Apply, Conditional, Conjugate, Form, Product, Zero
from ..nodes.arguments import map_argument
from ..rules.linearity import linear_fields
from ..rules.structural import application_retained, commute, structural_fields

if TYPE_CHECKING:
    from ..nodes.expressions import AbstractExpression


def _expand(node: Apply) -> AbstractExpression:
    f, arguments = node.form, node.arguments
    match f:
        case Form(body, entries):
            inherited = len(body.argument_spaces)
            supplied = dict(zip(entries, arguments[inherited:], strict=True))
            instantiated = partial(f, supplied) if entries else body
            return apply(instantiated, arguments[:inherited])
        case Conditional(predicate, left, right):
            return Conditional(predicate, apply(left, arguments), apply(right, arguments))
        case Product(first, second):
            n = len(first.argument_spaces)
            return multiply(apply(first, arguments[:n]), apply(second, arguments[n:]))
        case Conjugate(operand):
            return Conjugate(apply(operand, tuple(map_argument(Conjugate, g) for g in arguments)))
        case Zero():
            return zero_like(node, ())
        case _ if application_retained(f):
            return node
        case _:
            pass
    fields = linear_fields(f) or structural_fields(f)
    if fields is None:
        raise TypeError(f"no application rule for {type(f).__name__}")
    return commute(
        f,
        lambda operand: apply(operand, arguments),
        fields,
        arguments,
        lambda operand: not operand.argument_spaces,
    )


def expand_applications(f: AbstractExpression) -> AbstractExpression:
    return rewrite(f, lambda node: _expand(node) if isinstance(node, Apply) else node)


def applications_expanded(e: AbstractExpression) -> bool:
    return everywhere(e, lambda n: not isinstance(n, Apply) or application_retained(n.form))
