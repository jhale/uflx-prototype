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
from ..tensors import transpose
from ..tuples import each, members
from .commuting import commute


def derivative_of_node(f, d, constant, avoid):
    # The derivatives of the node classes, shared by every pass that differentiates.
    match f:
        case Product(b, c):
            return Sum(Product(d(b), c), Product(b, d(c)))
        case Division(b, c):
            return subtract(Division(d(b), c), Division(Product(b, d(c)), Product(c, c)))
        case Power(b, c) if constant(c):
            return Product(Product(c, Power(b, Sum(c, Integer(-1)))), d(b))
        case Power(b, c):
            return Product(f, Sum(Product(Logarithm(b), d(c)), Division(Product(c, d(b)), b)))
        case Absolute(b):
            return abs_derivative(b, d(b))
        case SquareRoot(b):
            return Division(d(b), Product(Integer(2), f))
        case Logarithm(b):
            return Division(d(b), b)
        case Determinant(A):
            from ..names import derived_indices
            from ..tensors import index_sum, indexed

            i, j = derived_indices("uflx.derivative", (f,), 2)
            return f * index_sum(indexed(Inverse(A), (i, j)) * indexed(d(A), (j, i)), (i, j))
        case Inverse(A):
            return inverse_derivative(A, d(A))
        case Action(l, p, w):
            if constant(l):
                return Action(l, p, each(d, w))
            if all(constant(v) for v in members(w)):
                return Action(d(l), p, w)
            return Sum(Action(d(l), p, w), Action(l, p, each(d, w)))
        case _ if f.commuting:
            return commute(f, d, avoid)
    raise TypeError(f"cannot occur: {f}")


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
