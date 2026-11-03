"""Experimental exact scalar collection, with no derivative-specific patterns.

Only rational/imaginary numeric factors are distributed over sums. Products of
nonconstant sums are not expanded. Scalar factors with no form arguments commute;
products carrying form arguments retain their order.
"""

from fractions import Fraction

from ..nodes import (
    AbstractExpression,
    Action,
    Conjugate,
    Division,
    Form,
    ImaginaryUnit,
    Integer,
    Product,
    Sum,
    Zero,
)
from ..static import complete_indices
from ..substitution import lowered, map_fields, memoized

ONE = (Fraction(1), Fraction(0))
ZERO = (Fraction(0), Fraction(0))


def plus(a, b):
    return a[0] + b[0], a[1] + b[1]


def times(a, b):
    return a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0]


def number(e):
    match e:
        case Integer(n):
            return Fraction(n), Fraction(0)
        case ImaginaryUnit():
            return Fraction(0), Fraction(1)
        case Division(Integer(n), Integer(d)) if d:
            return Fraction(n, d), Fraction(0)
        case Sum(a, b) | Product(a, b):
            left = number(a)
            if left is not None and (right := number(b)) is not None:
                return plus(left, right) if isinstance(e, Sum) else times(left, right)
    return None


def literal(value):
    def rational(q):
        n = Integer(q.numerator)
        return n if q.denominator == 1 else Division(n, Integer(q.denominator))

    real, imag = value
    if not imag:
        return rational(real)
    imaginary = ImaginaryUnit() if imag == 1 else Product(rational(imag), ImaginaryUnit())
    return imaginary if not real else Sum(rational(real), imaginary)


def product(factors):
    out = factors[0]
    for factor in factors[1:]:
        out = Product(out, factor)
    return out


def split_product(e):
    coefficient, factors = ONE, []

    def collect(node):
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


def scaled(coefficient, factors):
    if not factors:
        return literal(coefficient)
    body = product(factors)
    return body if coefficient == ONE else Product(literal(coefficient), body)


def collected(e):
    terms = {}

    def collect(node, scale):
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
        out = Sum(out, part)
    return out


def preserve(e, out):
    # Cancellation may remove indexed dependencies; retain the original family.
    if out.value_shape != e.value_shape or out.argument_spaces != e.argument_spaces:
        return e
    if e.scalar_type != out.scalar_type:
        out = Sum(out, Zero(e.value_shape, e.free_indices, e.argument_spaces, e.scalar_type))
    return complete_indices(e, out)


def operator_rule(e):
    # Use only contracts already supplied by core operations. No holomorphic test.
    if isinstance(e, Action) and not isinstance(e.argument, tuple):
        coefficient, factors = split_product(e.argument)
        if coefficient == ZERO:
            return Zero(e.value_shape, e.free_indices, e.argument_spaces, e.scalar_type)
        if coefficient != ONE:
            arg = scaled(ONE, factors)
            return Product(literal(coefficient), Action(e.form, e.position, arg))
    if isinstance(e, Product):
        coefficient, factors = split_product(e)
        if coefficient != ONE and len(factors) == 1 and isinstance(factors[0], Form):
            f = factors[0]
            return Form(Product(literal(coefficient), f.expression), f.arguments)
    if isinstance(e, Sum) and isinstance(e.first, Form) and isinstance(e.second, Form):
        if e.first.arguments == e.second.arguments:
            return Form(Sum(e.first.expression, e.second.expression), e.first.arguments)
    if isinstance(e, Conjugate):
        coefficient, factors = split_product(e.operand)
        if factors and coefficient != ONE:
            return Product(literal((coefficient[0], -coefficient[1])), Conjugate(product(factors)))
    return e


def rewrite(e):
    if not isinstance(e, AbstractExpression) or e.value_shape:
        return e
    if isinstance(e, Sum | Product):
        return preserve(e, collected(e))
    if isinstance(e, Conjugate):
        if isinstance(e.operand, Conjugate):
            return e.operand.operand
        if (value := number(e.operand)) is not None:
            return literal((value[0], -value[1]))
    return e


def normalizer():
    """Build a normalisation traversal with a memo local to one lowering call."""

    def walk(walk, e):
        new = map_fields(e, walk, lowered(e))
        moved = operator_rule(new)
        if moved is not new:
            return walk(preserve(new, moved))
        return rewrite(new)

    return memoized(walk)
