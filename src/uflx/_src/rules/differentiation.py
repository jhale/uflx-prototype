# Copyright (C) 2026 Jack S. Hale
#
# This file is part of uflx-core-experimental (https://fenics.github.io/uflx/language/)
#
# SPDX-License-Identifier:    MIT
"""Product, chain and matrix derivative rules."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from functools import cache
from typing import TYPE_CHECKING, TypeAlias

from ..algorithms.rewriting import complete_indices, promote, zero_like
from ..algorithms.substitution import occurs_in_values, substitute_values
from ..algorithms.traversal import memoized
from ..arithmetic import real_part, total
from ..nodes import (
    Absolute,
    AbstractExpression,
    Apply,
    Complex,
    Conditional,
    Conjugate,
    Determinant,
    DirectionalDerivative,
    Division,
    Float,
    Integer,
    Inverse,
    Logarithm,
    Power,
    Product,
    SquareRoot,
    Zero,
)
from ..nodes.arguments import argument_members, map_argument
from ..tensors import tensor, transpose
from .linearity import linear_fields
from .structural import commute, structural_fields

if TYPE_CHECKING:
    from ..nodes.domains import AbstractDomain
    from ..nodes.forms import Function
    from ..nodes.indices import Index
    from ..nodes.types import Construction


Transform: TypeAlias = Callable[[AbstractExpression], AbstractExpression]
DerivativeRule: TypeAlias = Callable[
    [AbstractExpression, Transform, Transform], AbstractExpression | None
]


def derivative_of_node(
    f: AbstractExpression,
    d: Callable[[AbstractExpression], AbstractExpression],
    constant: Callable[[AbstractExpression], bool],
    avoid: Sequence[Construction],
    at: Callable[[AbstractExpression], AbstractExpression] = lambda e: e,
) -> AbstractExpression:
    """D01-D10, D12, D19-D20, followed by structural and linear rules.

    d differentiates an operand; at evaluates it at the differentiation point.
    Spatial differentiation leaves at as the identity.
    """
    match f:
        case Conditional(predicate, left, right):
            return Conditional(at(predicate), d(left), d(right))
        case Apply(form, arguments):
            points = tuple(map_argument(at, slot) for slot in arguments)
            terms = [] if constant(form) else [Apply(d(form), points)]
            for i, slot in enumerate(arguments):
                if not all(constant(member) for member in argument_members(slot)):
                    varied = (*points[:i], map_argument(d, slot), *points[i + 1 :])
                    terms.append(Apply(at(form), varied))
            return total(terms) if terms else zero_like(f, ())
        case Product(b, c):
            return d(b) * at(c) + at(b) * d(c)
        case Division(b, c):
            return d(b) / at(c) - (at(b) * d(c)) / (at(c) * at(c))
        case Power(_, c) if _literal_value(c) == 0:
            # D19 precedes the general power rule, including at a zero base.
            return zero_like(f, f.argument_spaces)
        case Power(b, c) if _literal_value(c) == 1:
            # D20. The caller completes the derivative's type.
            return d(b)
        case Power(b, c) if constant(c):
            return at(c) * Power(at(b), at(c) + Integer(-1)) * d(b)
        case Power(b, c):
            return at(f) * (Logarithm(at(b)) * d(c) + (at(c) * d(b)) / at(b))
        case Absolute(b):
            value = at(b)
            return real_part(Conjugate(value) * d(b)) / Absolute(value)
        case SquareRoot(b):
            return d(b) / (Integer(2) * at(f))
        case Logarithm(b):
            return d(b) / at(b)
        case Determinant(a):
            n = a.value_shape[0]
            assert isinstance(n, int)
            if n == 0:
                return zero_like(f, f.argument_spaces)
            matrix_derivative = d(a)
            if n == 1:
                return matrix_derivative[0, 0]
            matrix = at(a)
            rows = tuple(matrix[r] for r in range(n))
            # The determinant is linear in each row, including at singular matrices.
            return total(
                Determinant(tensor((*rows[:r], matrix_derivative[r], *rows[r + 1 :])))
                for r in range(n)
            )
        case Inverse(a):
            matrix = at(a)
            derivative = d(a)
            inverse_matrix = Inverse(matrix)
            inverse_derivative = -(inverse_matrix @ derivative @ inverse_matrix)
            if matrix.value_shape[0] == matrix.value_shape[1]:
                return inverse_derivative
            # The rectangular case uses the derivative of the pseudo-inverse.
            transposed_derivative = transpose(derivative)
            correction = transposed_derivative - transposed_derivative @ matrix @ inverse_matrix
            return inverse_derivative + inverse_matrix @ transpose(inverse_matrix) @ correction
        case _:
            fields = linear_fields(f) or structural_fields(f)
            if fields is None:
                raise TypeError(f"no derivative rule for {type(f).__name__}")
            return commute(f, d, fields, avoid, constant, at)


def _literal_value(e: AbstractExpression) -> int | float | None:
    """Exact literal values for D19-D20, before point substitution."""
    match e:
        case Integer(n) | Float(n):
            return n
        case Complex(a, b) if b.value == 0:
            return a.value
        case Zero():
            return 0
        case _:
            return None


def deriv(
    f: AbstractExpression,
    x: Function | tuple[Function, ...],
    a: AbstractExpression | tuple[AbstractExpression, ...],
    h: AbstractExpression | tuple[AbstractExpression, ...],
    extension_rule: DerivativeRule | None = None,
) -> AbstractExpression:
    # Evaluate undifferentiated operands at the point; inserted directions stay unchanged.
    xs, points, directions = argument_members(x), argument_members(a), argument_members(h)
    points = tuple(promote(v, p) for v, p in zip(xs, points, strict=True))
    directions = tuple(promote(v, w) for v, w in zip(xs, directions, strict=True))
    mapping = dict(zip(xs, points, strict=True))

    @cache
    def at(e: AbstractExpression) -> AbstractExpression:
        return substitute_values(e, mapping)

    def constant(e: AbstractExpression) -> bool:
        return not any(occurs_in_values(v, e) for v in xs)

    def differentiate(
        d: Callable[[AbstractExpression], AbstractExpression], e: AbstractExpression
    ) -> AbstractExpression:
        expected = DirectionalDerivative(e, x, a, h)
        if constant(e):  # L10, SH15, D15.
            return zero_like(expected, expected.argument_spaces)
        for variable, direction in zip(xs, directions, strict=True):
            if e == variable:  # D11.
                return complete_indices(expected, direction)
        out = extension_rule(e, at, d) if extension_rule is not None else None
        if out is None:
            out = derivative_of_node(e, d, constant, [*xs, *points, *directions], at)
        return complete_indices(expected, out)

    return complete_indices(DirectionalDerivative(f, x, a, h), memoized(differentiate)(f))


def coordinate_partial(f: AbstractExpression, P: AbstractDomain, j: Index) -> AbstractExpression:
    """D17. The coordinate derivative on a full-dimensional domain."""
    from ..algorithms.names import derived_index
    from ..tensors import component_tensor, identity, indexed

    i = derived_index("uflx.apply_gradients", f, P)
    return component_tensor(indexed(identity(P.ambient_coordinate_domain.dimension), (i, j)), (i,))
