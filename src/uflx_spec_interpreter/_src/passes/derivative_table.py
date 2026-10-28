from ..arithmetic import subtract
from ..nodes import (
    Absolute,
    Action,
    Conjugate,
    Division,
    DualSpace,
    Function,
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
from .commuting import COMMUTING, commute


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
        case Inverse(A):
            return inverse_derivative(A, d(A))
        case Action(Function(space, _) as l, 0, w) if isinstance(space, DualSpace):
            return Action(l, 0, d(w))
        case _ if type(f) in COMMUTING:
            return commute(f, d, avoid)
    raise TypeError(f"cannot occur: {f}")


def abs_derivative(b, db):
    # Re(conj(b) db) / |b|, which is real arithmetic like the Absolute it replaces.
    return Division(RealPart(Product(Conjugate(b), db)), Absolute(b))


def inverse_derivative(A, dA):
    # The identity-free form -A⁺ dA A⁺ + A⁺ (A⁺)ᵀ (dAᵀ - dAᵀ A A⁺).
    Ai, dAt = Inverse(A), transpose(dA)
    return -(Ai @ dA @ Ai) + Ai @ transpose(Ai) @ (dAt - dAt @ A @ Ai)
