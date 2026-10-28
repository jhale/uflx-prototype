from ..nodes import (
    Absolute,
    Conjugate,
    DirectionalDerivative,
    Form,
    GateauxDerivative,
    ImaginaryPart,
    Num,
    Product,
    RealPart,
    Sum,
)
from ..static import arity, zero_like
from ..substitution import expressions, map_fields, occurs, operands

NONHOLOMORPHIC = (Conjugate, Absolute, RealPart, ImaginaryPart)


def rule(g):
    # The rule table of apply_gateaux_derivatives, keyed on the space of x and the entry.
    match g:
        case GateauxDerivative(F, x, a, entry) if not x.space.complex:
            return Form(DirectionalDerivative(F, x, a, direction(entry)), (entry,))
        case GateauxDerivative(F, x, _, Conjugate()) if holomorphic(F, x):
            return zero_like(F, arity(F) + 1)
        case GateauxDerivative(F, x, a, w) if holomorphic(F, x):
            return Form(DirectionalDerivative(F, x, a, w), (w,))
        case GateauxDerivative(F, x, a, Conjugate(w) as entry):
            return Form(half_sum(F, x, a, w, Num(1j)), (entry,))
        case GateauxDerivative(F, x, a, w):
            return Form(half_sum(F, x, a, w, Num(-1j)), (w,))
    return g


def direction(entry):
    match entry:
        case Conjugate(w):
            return w
    return entry


def holomorphic(F, x, inside=False):
    # Syntactic, so x under Conj, Abs, Re or Im makes F non-holomorphic.
    if inside and occurs(x, F):
        return False
    return all(holomorphic(y, x, inside or isinstance(F, NONHOLOMORPHIC)) for y in operands(F))


def half_sum(F, x, a, w, sign):
    return Product(
        Num(0.5),
        Sum(
            DirectionalDerivative(F, x, a, w),
            Product(sign, DirectionalDerivative(F, x, a, Product(Num(1j), w))),
        ),
    )


def apply_gateaux_derivatives(e):
    # Inner derivatives first, then the rule for each Gateaux derivative.
    return rule(map_fields(e, apply_gateaux_derivatives))


def in_n1(e):
    return not isinstance(e, GateauxDerivative) and all(in_n1(c) for c in expressions(e))
