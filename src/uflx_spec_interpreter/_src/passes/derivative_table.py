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
from ..substitution import freshen, map_fields
from ..tensors import transpose
from ..tuples import each, members


def derivative_of_node(f, d, constant, avoid, at=lambda e: e):
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
        case Determinant(A):
            from ..names import derived_indices
            from ..tensors import index_sum, indexed

            i, j = derived_indices("uflx.derivative", (f,), 2)
            return at(f) * index_sum(
                indexed(Inverse(at(A)), (i, j)) * indexed(d(A), (j, i)), (i, j)
            )
        case Inverse(A):
            return inverse_derivative(at(A), d(A))
        case Action(l, p, w):
            if constant(l):
                return Action(at(l), p, each(d, w))
            if all(constant(v) for v in members(w)):
                return Action(d(l), p, each(at, w))
            return Sum(Action(d(l), p, each(at, w)), Action(at(l), p, each(d, w)))
        case _ if f.commuting:
            other = tuple(n for n in f.value_fields if n not in f.commuting)
            return map_fields(map_fields(f, at, other), d, f.commuting)
    raise TypeError(f"no derivative rule for {type(f).__name__}")


def abs_derivative(b, db):
    # Re(conj(b) db) / |b|, which is real arithmetic like the Absolute it replaces.
    return Division(RealPart(Product(Conjugate(b), db)), Absolute(b))


def inverse_derivative(A, dA):
    # The identity-free form -A⁺ dA A⁺ + A⁺ (A⁺)ᵀ (dAᵀ - dAᵀ A A⁺).
    Ai = Inverse(A)
    if A.value_shape[0] == A.value_shape[1]:
        return -(Ai @ dA @ Ai)
    dAt = transpose(dA)
    return -(Ai @ dA @ Ai) + Ai @ transpose(Ai) @ (dAt - dAt @ A @ Ai)
