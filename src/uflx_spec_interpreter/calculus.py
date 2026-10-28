from .algebra import matmul, minus, negate, transpose
from .commuting import COMMUTING, commute
from .nodes import Abs, Action, Conj, Div, Fn, Inv, Log, Num, Power, Prod, Sqrt, Sum


def derivative_of_node(f, d, constant, avoid):
    # The derivatives of the node classes, shared by every pass that differentiates.
    match f:
        case Prod(b, c):
            return Sum(Prod(d(b), c), Prod(b, d(c)))
        case Div(b, c):
            return minus(Div(d(b), c), Div(Prod(b, d(c)), Prod(c, c)))
        case Power(b, c) if constant(c):
            return Prod(Prod(c, Power(b, Sum(c, Num(-1)))), d(b))
        case Power(b, c):
            return Prod(f, Sum(Prod(Log(b), d(c)), Div(Prod(c, d(b)), b)))
        case Abs(b):
            return abs_derivative(b, d(b))
        case Sqrt(b):
            return Div(d(b), Prod(Num(2), f))
        case Log(b):
            return Div(d(b), b)
        case Inv(A):
            return inverse_derivative(A, d(A))
        case Action(Fn(space, _) as l, 0, w) if space.dual:
            return Action(l, 0, d(w))
        case _ if type(f) in COMMUTING:
            return commute(f, d, avoid)
    raise TypeError(f"cannot occur: {f}")


def abs_derivative(b, db):
    return Div(Sum(Prod(Conj(b), db), Prod(b, Conj(db))), Prod(Num(2), Abs(b)))


def inverse_derivative(A, dA):
    # The identity-free form -A⁺ dA A⁺ + A⁺ (A⁺)ᵀ (dAᵀ - dAᵀ A A⁺).
    Ai, dAt = Inv(A), transpose(dA)
    return Sum(
        negate(matmul(Ai, dA, Ai)), matmul(Ai, transpose(Ai), Sum(dAt, negate(matmul(dAt, A, Ai))))
    )
