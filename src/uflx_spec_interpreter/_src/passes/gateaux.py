from ..nodes import (
    Absolute,
    Conjugate,
    DirectionalDerivative,
    Division,
    Form,
    GateauxDerivative,
    ImaginaryPart,
    ImaginaryUnit,
    Integer,
    Product,
    RealPart,
    ScalarType,
    Sum,
)
from ..static import zero_like
from ..substitution import bottom_up, everywhere, memoized, occurs, operands
from ..tuples import members

NONHOLOMORPHIC = (Conjugate, Absolute, RealPart, ImaginaryPart)


def rule(g):
    # The rule table of apply_gateaux_derivatives, keyed on the space of x and the entry.
    match g:
        case GateauxDerivative(F, x, a, entry) if real(x):
            return Form(DirectionalDerivative(F, x, a, direction(entry)), (entry,))
        case GateauxDerivative(F, x, _, entry) if antilinear(entry) and holomorphic(F, x):
            return zero_like(F, g.argument_spaces)
        case GateauxDerivative(F, x, a, w) if holomorphic(F, x):
            return Form(DirectionalDerivative(F, x, a, w), (w,))
        case GateauxDerivative(F, x, a, entry) if antilinear(entry):
            return Form(half_sum(F, x, a, direction(entry), ImaginaryUnit()), (entry,))
        case GateauxDerivative(F, x, a, w):
            return Form(half_sum(F, x, a, w, Product(Integer(-1), ImaginaryUnit())), (w,))
    return g


def real(x):
    # A tuple variable is real only if every member is, the scalar type of its derivative.
    return all(m.space.scalar_type == ScalarType.real for m in members(x))


def antilinear(entry):
    return all(isinstance(m, Conjugate) for m in members(entry))


def direction(entry):
    # The function of an entry, member by member for a tuple.
    if isinstance(entry, tuple):
        return tuple(map(direction, entry))
    return entry.operand if isinstance(entry, Conjugate) else entry


def holomorphic(F, x):
    # Syntactic, so a member of x under Conj, Abs, Re or Im makes F non-holomorphic.
    def walk(walk, G, inside):
        if inside and any(occurs(m, G) for m in members(x)):
            return False
        return all(walk(y, inside or isinstance(G, NONHOLOMORPHIC)) for y in operands(G))

    return memoized(walk)(F, False)


def times_i(w):
    if isinstance(w, tuple):
        return tuple(map(times_i, w))
    return Product(ImaginaryUnit(), w)


def half_sum(F, x, a, w, sign):
    return Product(
        Division(Integer(1), Integer(2)),
        Sum(
            DirectionalDerivative(F, x, a, w),
            Product(sign, DirectionalDerivative(F, x, a, times_i(w))),
        ),
    )


def apply_gateaux_derivatives(f):
    # Inner derivatives first, then the rule for each Gateaux derivative.
    return bottom_up(f, rule)


def in_n1(e):
    return everywhere(e, lambda n: not isinstance(n, GateauxDerivative))
