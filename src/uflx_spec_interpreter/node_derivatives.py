from .algebra import matrix_product, negate, subtract, transpose
from .commuting import COMMUTING, commute
from .nodes import (
    Absolute,
    Action,
    Conjugate,
    Division,
    Function,
    Inverse,
    Logarithm,
    Num,
    Power,
    Product,
    SquareRoot,
    Sum,
)


def derivative_of_node(f, d, constant, avoid):
    # The derivatives of the node classes, shared by every pass that differentiates.
    match f:
        case Product(b, c):
            return Sum(Product(d(b), c), Product(b, d(c)))
        case Division(b, c):
            return subtract(Division(d(b), c), Division(Product(b, d(c)), Product(c, c)))
        case Power(b, c) if constant(c):
            return Product(Product(c, Power(b, Sum(c, Num(-1)))), d(b))
        case Power(b, c):
            return Product(f, Sum(Product(Logarithm(b), d(c)), Division(Product(c, d(b)), b)))
        case Absolute(b):
            return abs_derivative(b, d(b))
        case SquareRoot(b):
            return Division(d(b), Product(Num(2), f))
        case Logarithm(b):
            return Division(d(b), b)
        case Inverse(A):
            return inverse_derivative(A, d(A))
        case Action(Function(space, _) as l, 0, w) if space.dual:
            return Action(l, 0, d(w))
        case _ if type(f) in COMMUTING:
            return commute(f, d, avoid)
    raise TypeError(f"cannot occur: {f}")


def abs_derivative(b, db):
    return Division(
        Sum(Product(Conjugate(b), db), Product(b, Conjugate(db))), Product(Num(2), Absolute(b))
    )


def inverse_derivative(A, dA):
    # The identity-free form -A⁺ dA A⁺ + A⁺ (A⁺)ᵀ (dAᵀ - dAᵀ A A⁺).
    Ai, dAt = Inverse(A), transpose(dA)
    return Sum(
        negate(matrix_product(Ai, dA, Ai)),
        matrix_product(Ai, transpose(Ai), Sum(dAt, negate(matrix_product(dAt, A, Ai)))),
    )
