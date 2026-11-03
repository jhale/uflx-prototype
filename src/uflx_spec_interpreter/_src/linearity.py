"""Conservative shared certificates of dependence on free function variables.

Certificates are sufficient, not necessary: cancellation can hide linearity.
Unknown never means nonlinear. Zero satisfies both complex linearity conventions.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Literal

from .nodes import (
    AbstractExpression,
    Action,
    ComponentTensor,
    Conjugate,
    Curl,
    Divergence,
    Division,
    Form,
    Function,
    Gradient,
    ImaginaryPart,
    Indexed,
    IndexSum,
    Integer,
    Integral,
    Product,
    RealPart,
    ScalarType,
    Sum,
    Tensor,
    Zero,
)
from .nodes.forms import entry_function
from .substitution import memoized, occurs
from .tuples import members

ZERO: frozenset[tuple[int, int]] = frozenset()
CONSTANT = frozenset({(0, 0)})
LINEAR = frozenset({(1, 0)})
ANTILINEAR = frozenset({(0, 1)})


Dependence = Literal[
    "zero",
    "independent",
    "complex-linear",
    "complex-antilinear",
    "real-linear",
    "affine",
    "unknown",
]


def dependence(
    expression: AbstractExpression, variables: Function | tuple[Function, ...]
) -> Dependence:
    """Certify dependence on free functions, treating other data as fixed.

    Args:
        expression: A valid symbolic expression, possibly form-valued.
        variables: One function or a nonempty tuple of distinct functions.
            Tuple dependence is joint: u*v is linear in u alone, but is not
            certified linear in (u, v). Singleton tuples are allowed here.

    Returns:
        A sufficient certificate, not an exhaustive classification:
        ``zero`` certifies identically zero; ``independent`` certifies no free
        dependence; ``complex-linear`` and ``complex-antilinear`` certify the
        respective homogeneous dependence; ``real-linear`` certifies additive,
        real-homogeneous dependence; ``affine`` permits an independent term
        plus a real-linear part; ``unknown`` gives no conclusion.

        Results need not be the strongest possible: cancellation is not
        performed. Real-linear may also be complex-linear; independent may
        also be zero. Queries involving real variables only claim real-linearity.
        Zero satisfies both complex conventions. Bound occurrences do not count
        as free dependence, and coefficients may have free family indices.

    Raises:
        TypeError: The variable tuple is empty.
        ValueError: A variable is repeated.

    Certificates assume existing node contracts, including valid form slots,
    fixed geometry and defined coefficients. They do not prove global
    definedness (for example, a denominator may vanish), differentiability or
    the validity of declarations inside opaque forms. Unknown must not be used
    as a verdict that a form is invalid or nonlinear. No nodes are rewritten.
    """
    variables = tuple(members(variables))
    if not variables:
        raise TypeError("dependence variables must be one or more functions")
    if len(set(variables)) != len(variables):
        raise ValueError("dependence variables must be distinct")
    real_variables = any(v.scalar_type == ScalarType.real for v in variables)

    def union(
        parts: Iterable[frozenset[tuple[int, int]] | None],
    ) -> frozenset[tuple[int, int]] | None:
        parts = list(parts)
        return (
            None
            if any(p is None for p in parts)
            else frozenset(pair for part in parts if part is not None for pair in part)
        )

    def degrees(
        walk: Callable[[AbstractExpression], frozenset[tuple[int, int]] | None],
        e: AbstractExpression,
    ) -> frozenset[tuple[int, int]] | None:
        if isinstance(e, Zero) or (isinstance(e, Integer) and e.value == 0):
            return ZERO
        if e in variables:
            return LINEAR
        if not any(occurs(v, e) for v in variables):
            return CONSTANT
        match e:
            case Sum(a, b):
                return union((walk(a), walk(b)))
            case Product(a, b):
                left, right = walk(a), walk(b)
                if left == ZERO or right == ZERO:
                    return ZERO
                if left is None or right is None:
                    return None
                # Cap degrees: only constant and degree-one certificates are needed.
                return frozenset((min(2, p + r), min(2, q + s)) for p, q in left for r, s in right)
            case Division(a, b) if not any(occurs(v, b) for v in variables):
                return walk(a)
            case Conjugate(a):
                result = walk(a)
                return None if result is None else frozenset((q, p) for p, q in result)
            case RealPart(a) | ImaginaryPart(a):
                result = walk(a)
                return None if result is None else result | frozenset((q, p) for p, q in result)
            case Tensor(entries):
                return union(walk(a) for a in entries)
            case Gradient() | Divergence() | Curl() | Integral() | ComponentTensor() | IndexSum():
                return walk(e.expression)
            case Indexed():
                # Gather indices can themselves carry expression dependencies.
                if any(
                    any(occurs(v, i) for v in variables)
                    for i in e.indices
                    if not isinstance(i, int)
                ):
                    return None
                return walk(e.expression)
            case Form():
                # A bound variable is not a free dependency of this form value.
                if any(v in e.bound() for v in variables):
                    return None
                return walk(e.expression)
            case Action():
                if any(occurs(v, e.form) for v in variables):
                    return None
                return union(walk(a) for a in members(e.argument))
            case _:
                return None

    result = memoized(degrees)(expression)
    if result == ZERO:
        return "zero"
    if result == CONSTANT:
        return "independent"
    if real_variables and result is not None and result <= LINEAR | ANTILINEAR:
        return "real-linear"
    if result == LINEAR:
        return "complex-linear"
    if result == ANTILINEAR:
        return "complex-antilinear"
    if result is not None and result <= LINEAR | ANTILINEAR:
        return "real-linear"
    if result is not None and result <= CONSTANT | LINEAR | ANTILINEAR:
        return "affine"
    return "unknown"


def argument_dependence(form: Form, position: int) -> Dependence:
    """Inspect an entry introduced by an explicit Form, without validating it.

    Args:
        form: An explicit Form with valid binding entries.
        position: Zero-based index into form.arguments, excluding any argument
            slots inherited from form.expression. Negative indices are rejected.

    Returns:
        The dependence certificate for the underlying function(s) of the entry.
        For complex variables, a plain entry expects complex-linear or zero;
        a conjugate entry expects complex-antilinear or zero. For real variables,
        either convention expects real-linear or zero. Unknown is inconclusive.
        A mixed plain/conjugate tuple is analysed jointly in its underlying
        functions, not componentwise against each entry's convention; this
        query alone does not validate such a declaration.

    Raises:
        TypeError: The position is not an exact int.
        ValueError: position is out of range or the entry repeats a function.

    Unlike dependence(form, v), this query inspects the binding body: the chosen
    entry is exposed, while any inner binding still shadows its functions.
    """
    if type(position) is not int:
        raise TypeError("argument position must be an int")
    if not 0 <= position < len(form.arguments):
        raise ValueError("argument position is outside the explicit binding entries")
    entry = form.arguments[position]
    return dependence(form.expression, tuple(entry_function(v) for v in members(entry)))
