from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

from ..arithmetic import subtract
from ..nodes import (
    Absolute,
    Action,
    Conjugate,
    Determinant,
    Division,
    Integer,
    Inverse,
    Logarithm,
    Power,
    Product,
    RealPart,
    SquareRoot,
    Sum,
)
from ..substitution import freshen, map_values
from ..tensors import transpose
from ..tuples import each, members

if TYPE_CHECKING:
    from ..nodes.base import AbstractExpression
    from ..types import Construction


def derivative_of_node(
    f: AbstractExpression,
    d: Callable[[AbstractExpression], AbstractExpression],
    constant: Callable[[AbstractExpression], bool],
    avoid: Sequence[Construction],
    at: Callable[[AbstractExpression], AbstractExpression] = lambda e: e,
) -> AbstractExpression:
    # The derivatives of the node classes, shared by every pass that differentiates.
    f = freshen(f, avoid)
    match f:
        case Product(b, c):
            return Sum(Product(d(b), at(c)), Product(at(b), d(c)))
        case Division(b, c):
            return subtract(
                Division(d(b), at(c)), Division(Product(at(b), d(c)), Product(at(c), at(c)))
            )
        case Power(b, c) if constant(c):
            return Product(Product(at(c), Power(at(b), Sum(at(c), Integer(-1)))), d(b))
        case Power(b, c):
            return Product(
                at(f), Sum(Product(Logarithm(at(b)), d(c)), Division(Product(at(c), d(b)), at(b)))
            )
        case Absolute(b):
            return abs_derivative(at(b), d(b))
        case SquareRoot(b):
            return Division(d(b), Product(Integer(2), at(f)))
        case Logarithm(b):
            return Division(d(b), at(b))
        case Determinant(a):
            from ..names import derived_indices
            from ..tensors import index_sum, indexed

            i, j = derived_indices("uflx.derivative", (f,), 2)
            return at(f) * index_sum(
                indexed(Inverse(at(a)), (i, j)) * indexed(d(a), (j, i)), (i, j)
            )
        case Inverse(a):
            return inverse_derivative(at(a), d(a))
        case Action(l, p, w):
            if constant(l):
                return Action(at(l), p, each(d, w))
            if all(constant(v) for v in members(w)):
                return Action(d(l), p, each(at, w))
            return Sum(Action(d(l), p, each(at, w)), Action(at(l), p, each(d, w)))
        case _ if f.commuting:
            other = tuple(n for n in f.value_fields if n not in f.commuting)
            return map_values(map_values(f, at, other), d, f.commuting)
        case _:
            raise TypeError(f"no derivative rule for {type(f).__name__}")


def abs_derivative(b: AbstractExpression, db: AbstractExpression) -> AbstractExpression:
    # Re(conj(b) db) / |b|, which is real arithmetic like the Absolute it replaces.
    return Division(RealPart(Product(Conjugate(b), db)), Absolute(b))


def inverse_derivative(A: AbstractExpression, dA: AbstractExpression) -> AbstractExpression:
    # The identity-free form -A⁺ dA A⁺ + A⁺ (A⁺)ᵀ (dAᵀ - dAᵀ A A⁺).
    Ai = Inverse(A)
    if A.value_shape[0] == A.value_shape[1]:
        return -(Ai @ dA @ Ai)
    dAt = transpose(dA)
    return -(Ai @ dA @ Ai) + Ai @ transpose(Ai) @ (dAt - dAt @ A @ Ai)
