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
from ..static import complete_indices
from ..substitution import lowered, map_values, memoized

ONE = (Fraction(1), Fraction(0))
ZERO = (Fraction(0), Fraction(0))


def plus(a: tuple[Fraction, Fraction], b: tuple[Fraction, Fraction]) -> tuple[Fraction, Fraction]:
    return a[0] + b[0], a[1] + b[1]


def times(a: tuple[Fraction, Fraction], b: tuple[Fraction, Fraction]) -> tuple[Fraction, Fraction]:
    return a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0]


def number(e: AbstractExpression) -> tuple[Fraction, Fraction] | None:
    match e:
        case Integer(n):
            return Fraction(n), Fraction(0)
        case Complex(a, b):
            return Fraction(a.value), Fraction(b.value)
        case Division(Integer(n), Integer(d)) if d:
            return Fraction(n, d), Fraction(0)
        case Sum(a, b) | Product(a, b):
            left = number(a)
            if left is not None and (right := number(b)) is not None:
                return plus(left, right) if isinstance(e, Sum) else times(left, right)
        case _:
            return None


def literal(value: tuple[Fraction, Fraction]) -> AbstractExpression:
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


def product(factors: Sequence[AbstractExpression]) -> AbstractExpression:
    out = factors[0]
    for factor in factors[1:]:
        out = multiply(out, factor)
    return out


def split_product(
    e: AbstractExpression,
) -> tuple[tuple[Fraction, Fraction], tuple[AbstractExpression, ...]]:
    coefficient = ONE
    factors: list[AbstractExpression] = []

    def collect(node: AbstractExpression) -> None:
        nonlocal coefficient
        if isinstance(node, Zero):
            coefficient = ZERO
        elif (value := number(node)) is not None:
            coefficient = times(coefficient, value)
        elif isinstance(node, Product):
            collect(node.first)
            collect(node.second)
        else:
            factors.append(node)

    collect(e)
    if all(not f.argument_spaces and not f.value_shape for f in factors):
        factors.sort(key=lambda f: f.digest)
    return coefficient, tuple(factors)


def scaled(
    coefficient: tuple[Fraction, Fraction], factors: tuple[AbstractExpression, ...]
) -> AbstractExpression:
    if not factors:
        return literal(coefficient)
    body = product(factors)
    return body if coefficient == ONE else multiply(literal(coefficient), body)


def collected(e: AbstractExpression) -> AbstractExpression:
    terms: dict[tuple[AbstractExpression, ...], tuple[Fraction, Fraction]] = {}

    def collect(node: AbstractExpression, scale: tuple[Fraction, Fraction]) -> None:
        if isinstance(node, Sum):
            collect(node.first, scale)
            collect(node.second, scale)
            return
        coefficient, factors = split_product(node)
        coefficient = times(coefficient, scale)
        if len(factors) == 1 and isinstance(factors[0], Sum):
            collect(factors[0], coefficient)
        else:
            terms[factors] = plus(terms.get(factors, ZERO), coefficient)

    collect(e, ONE)
    parts = [scaled(c, fs) for fs, c in terms.items() if c != ZERO]
    if not parts:
        return Zero(e.value_shape, e.free_indices, e.argument_spaces, e.scalar_type)
    if not e.argument_spaces:
        parts.sort(key=lambda x: x.digest)
    out = parts[0]
    for part in parts[1:]:
        out = add(out, part)
    return out


def preserve(e: AbstractExpression, out: AbstractExpression) -> AbstractExpression:
    return complete_indices(e, out)


def operator_rule(e: AbstractExpression) -> AbstractExpression:
    # Use only contracts already supplied by core operations. No holomorphic test.
    if isinstance(e, Action) and not isinstance(e.argument, tuple):
        coefficient, factors = split_product(e.argument)
        if coefficient == ZERO:
            return Zero(e.value_shape, e.free_indices, e.argument_spaces, e.scalar_type)
        if coefficient != ONE:
            arg = scaled(ONE, factors)
            return multiply(literal(coefficient), Action(e.form, e.position, arg))
    if isinstance(e, Product):
        coefficient, factors = split_product(e)
        if coefficient != ONE and len(factors) == 1 and isinstance(factors[0], Form):
            f = factors[0]
            return Form(multiply(literal(coefficient), f.expression), f.arguments)
    if isinstance(e, Sum) and isinstance(e.first, Form) and isinstance(e.second, Form):
        if e.first.arguments == e.second.arguments:
            return Form(add(e.first.expression, e.second.expression), e.first.arguments)
    if isinstance(e, Conjugate):
        coefficient, factors = split_product(e.operand)
        if factors and coefficient != ONE:
            return multiply(literal((coefficient[0], -coefficient[1])), Conjugate(product(factors)))
    return e


def rewrite(e: AbstractExpression) -> AbstractExpression:
    if e.value_shape:
        return e
    if isinstance(e, Sum | Product):
        return preserve(e, collected(e))
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
        if (value := number(e.operand)) is not None:
            return literal((value[0], -value[1]))
    return e


def normalizer() -> Callable[[AbstractExpression], AbstractExpression]:
    """Build a normalisation traversal with a memo local to one lowering call."""

    def walk(
        walk: Callable[[AbstractExpression], AbstractExpression], e: AbstractExpression
    ) -> AbstractExpression:
        new = map_values(e, walk, lowered(e))
        moved = operator_rule(new)
        if moved is not new:
            return walk(preserve(new, moved))
        return preserve(new, rewrite(new))

    return memoized(walk)
