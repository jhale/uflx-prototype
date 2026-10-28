from algebra import matmul, minus, negate, transpose
from core import COMMUTING, D, Abs, Action, Conj, Div, Fn, Inv, Log, Num, Power, Prod, Sqrt, Sum, commute, map_fields, occurs, operands, substitute
from static import arity, zero_like


def deriv(f, x, a, h):
    # The rules give the derivative at x, so the point is substituted for x afterwards.
    out = rules(f, x, a, h)
    return out if a == x else substitute(out, {x: a})


def rules(f, x, a, h):
    # The rule table of apply_derivatives, keyed on the expression f.
    d = lambda b: rules(b, x, a, h)
    match f:
        case _ if not occurs(x, f):
            return zero_like(f, arity(f), a, h)
        case _ if f == x:
            return h
        case Prod(b, c):
            return Sum(Prod(d(b), c), Prod(b, d(c)))
        case Div(b, c):
            return minus(Div(d(b), c), Div(Prod(b, d(c)), Prod(c, c)))
        case Power(b, c) if not occurs(x, c):
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
            return commute(f, d, [x, a, h])
    raise TypeError(f"cannot occur: {f}")


def abs_derivative(b, db):
    return Div(Sum(Prod(Conj(b), db), Prod(b, Conj(db))), Prod(Num(2), Abs(b)))


def inverse_derivative(A, dA):
    # The identity-free form -A⁺ dA A⁺ + A⁺ (A⁺)ᵀ (dAᵀ - dAᵀ A A⁺).
    Ai, dAt = Inv(A), transpose(dA)
    return Sum(negate(matmul(Ai, dA, Ai)), matmul(Ai, transpose(Ai), Sum(dAt, negate(matmul(dAt, A, Ai)))))


def apply_derivatives(e):
    # Inner derivatives first, then the rule for the expression of each derivative.
    match map_fields(e, apply_derivatives):
        case D(f, x, a, h):
            return deriv(f, x, a, h)
        case e:
            return e


def in_n3(e):
    return not isinstance(e, D) and all(in_n3(c) for c in operands(e))
