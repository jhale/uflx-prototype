from ..nodes import Abs, Conj, D, Form, Gateaux, Im, Num, Prod, Re, Sum
from ..static import arity, zero_like
from ..substitution import map_fields, occurs, operands

NONHOLOMORPHIC = (Conj, Abs, Re, Im)


def rule(g):
    # The rule table of apply_gateaux_derivatives, keyed on the space of x and the entry.
    match g:
        case Gateaux(F, x, a, entry) if not x.space.complex:
            return Form(D(F, x, a, direction(entry)), (entry,))
        case Gateaux(F, x, _, Conj()) if holomorphic(F, x):
            return zero_like(F, arity(F) + 1)
        case Gateaux(F, x, a, w) if holomorphic(F, x):
            return Form(D(F, x, a, w), (w,))
        case Gateaux(F, x, a, Conj(w) as entry):
            return Form(half_sum(F, x, a, w, Num(1j)), (entry,))
        case Gateaux(F, x, a, w):
            return Form(half_sum(F, x, a, w, Num(-1j)), (w,))
    return g


def direction(entry):
    match entry:
        case Conj(w):
            return w
    return entry


def holomorphic(F, x, inside=False):
    # Syntactic, so x under Conj, Abs, Re or Im makes F non-holomorphic.
    if inside and occurs(x, F):
        return False
    return all(holomorphic(y, x, inside or isinstance(F, NONHOLOMORPHIC)) for y in operands(F))


def half_sum(F, x, a, w, sign):
    return Prod(Num(0.5), Sum(D(F, x, a, w), Prod(sign, D(F, x, a, Prod(Num(1j), w)))))


def apply_gateaux(e):
    # Inner derivatives first, then the rule for each Gateaux derivative.
    return rule(map_fields(e, apply_gateaux))


def in_n1(e):
    return not isinstance(e, Gateaux) and all(in_n1(c) for c in operands(e))
