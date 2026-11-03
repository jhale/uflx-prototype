"""Experimental exact scalar collection, with no derivative-specific patterns.

Only rational/imaginary numeric factors are distributed over sums. Products of
nonconstant sums are not expanded. Scalar factors with no form arguments commute;
products carrying form arguments retain their order.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from fractions import Fraction

from ..arithmetic import add, divide, multiply
from ..nodes import (
    AbstractExpression,
    Action,
    Complex,
    Conjugate,
    Division,
    Float,
    Form,
    Integer,
    Product,
    Sum,
    Zero,
)
from ..static import _complete_indices
from ..substitution import _lowered, _map_values, _memoized

_ONE = (Fraction(1), Fraction(0))
_ZERO = (Fraction(0), Fraction(0))


# Shared implementation helpers; the package public API is declared in __init__.py.
__all__ = ["_normalizer"]


def _plus(a: tuple[Fraction, Fraction], b: tuple[Fraction, Fraction]) -> tuple[Fraction, Fraction]:
    return a[0] + b[0], a[1] + b[1]


def _times(a: tuple[Fraction, Fraction], b: tuple[Fraction, Fraction]) -> tuple[Fraction, Fraction]:
    return a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0]


def _number(e: AbstractExpression) -> tuple[Fraction, Fraction] | None:
    match e:
        case Integer(n):
            return Fraction(n), Fraction(0)
        case Complex(a, b):
            return Fraction(a.value), Fraction(b.value)
        case Division(Integer(n), Integer(d)) if d:
            return Fraction(n, d), Fraction(0)
        case Sum(a, b) | Product(a, b):
            left = _number(a)
            if left is not None and (right := _number(b)) is not None:
                return _plus(left, right) if isinstance(e, Sum) else _times(left, right)
        case _:
            return None


def _literal(value: tuple[Fraction, Fraction]) -> AbstractExpression:
    def rational(q: Fraction) -> AbstractExpression:
        n = Integer(q.numerator)
        return n if q.denominator == 1 else divide(n, Integer(q.denominator))

    real, imag = value
    if not imag:
        return rational(real)
    imaginary = (
        Complex(Integer(0), Integer(1))
        if imag == 1
        else multiply(rational(imag), Complex(Integer(0), Integer(1)))
    )
    return imaginary if not real else add(rational(real), imaginary)


def _product(factors: Sequence[AbstractExpression]) -> AbstractExpression:
    out = factors[0]
    for factor in factors[1:]:
        out = multiply(out, factor)
    return out


def _split_product(
    e: AbstractExpression,
) -> tuple[tuple[Fraction, Fraction], tuple[AbstractExpression, ...]]:
    coefficient = _ONE
    factors: list[AbstractExpression] = []

    def collect(node: AbstractExpression) -> None:
        nonlocal coefficient
        if isinstance(node, Zero):
            coefficient = _ZERO
        elif (value := _number(node)) is not None:
            coefficient = _times(coefficient, value)
        elif isinstance(node, Product):
            collect(node.first)
            collect(node.second)
        else:
            factors.append(node)

    collect(e)
    if all(not f.argument_spaces and not f.value_shape for f in factors):
        factors.sort(key=lambda f: f.digest)
    return coefficient, tuple(factors)


def _scaled(
    coefficient: tuple[Fraction, Fraction], factors: tuple[AbstractExpression, ...]
) -> AbstractExpression:
    if not factors:
        return _literal(coefficient)
    body = _product(factors)
    return body if coefficient == _ONE else multiply(_literal(coefficient), body)


def _collected(e: AbstractExpression) -> AbstractExpression:
    terms: dict[tuple[AbstractExpression, ...], tuple[Fraction, Fraction]] = {}

    def collect(node: AbstractExpression, scale: tuple[Fraction, Fraction]) -> None:
        if isinstance(node, Sum):
            collect(node.first, scale)
            collect(node.second, scale)
            return
        coefficient, factors = _split_product(node)
        coefficient = _times(coefficient, scale)
        if len(factors) == 1 and isinstance(factors[0], Sum):
            collect(factors[0], coefficient)
        else:
            terms[factors] = _plus(terms.get(factors, _ZERO), coefficient)

    collect(e, _ONE)
    parts = [_scaled(c, fs) for fs, c in terms.items() if c != _ZERO]
    if not parts:
        return Zero(e.value_shape, e.free_indices, e.argument_spaces, e.scalar_type)
    if not e.argument_spaces:
        parts.sort(key=lambda x: x.digest)
    out = parts[0]
    for part in parts[1:]:
        out = add(out, part)
    return out


def _preserve(e: AbstractExpression, out: AbstractExpression) -> AbstractExpression:
    return _complete_indices(e, out)


def _operator_rule(e: AbstractExpression) -> AbstractExpression:
    # Use only contracts already supplied by core operations. No holomorphic test.
    if isinstance(e, Action) and not isinstance(e.argument, tuple):
        coefficient, factors = _split_product(e.argument)
        if coefficient == _ZERO:
            return Zero(e.value_shape, e.free_indices, e.argument_spaces, e.scalar_type)
        if coefficient != _ONE:
            arg = _scaled(_ONE, factors)
            return multiply(_literal(coefficient), Action(e.form, e.position, arg))
    if isinstance(e, Product):
        coefficient, factors = _split_product(e)
        if coefficient != _ONE and len(factors) == 1 and isinstance(factors[0], Form):
            f = factors[0]
            return Form(multiply(_literal(coefficient), f.expression), f.arguments)
    if isinstance(e, Sum) and isinstance(e.first, Form) and isinstance(e.second, Form):
        if e.first.arguments == e.second.arguments:
            return Form(add(e.first.expression, e.second.expression), e.first.arguments)
    if isinstance(e, Conjugate):
        coefficient, factors = _split_product(e.operand)
        if factors and coefficient != _ONE:
            return multiply(
                _literal((coefficient[0], -coefficient[1])), Conjugate(_product(factors))
            )
    return e


def _rewrite(e: AbstractExpression) -> AbstractExpression:
    if e.value_shape:
        return e
    if isinstance(e, Sum | Product):
        return _preserve(e, _collected(e))
    if isinstance(e, Conjugate):
        if isinstance(e.operand, Complex):
            a, b = e.operand.real, e.operand.imag
            # Keep signed zero: it selects a side of a complex branch cut.
            imag = (
                Float(-float(b.value))
                if b.value == 0
                else Integer(-b.value)
                if isinstance(b, Integer)
                else Float(-b.value)
            )
            return Complex(a, imag)
        if isinstance(e.operand, Conjugate):
            return e.operand.operand
        if (value := _number(e.operand)) is not None:
            return _literal((value[0], -value[1]))
    return e


def _normalizer() -> Callable[[AbstractExpression], AbstractExpression]:
    """Build a normalisation traversal with a memo local to one lowering call."""

    def walk(
        walk: Callable[[AbstractExpression], AbstractExpression], e: AbstractExpression
    ) -> AbstractExpression:
        new = _map_values(e, walk, _lowered(e))
        moved = _operator_rule(new)
        if moved is not new:
            return walk(_preserve(new, moved))
        return _preserve(new, _rewrite(new))

    return _memoized(walk)
