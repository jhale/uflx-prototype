from ..algebra import total
from ..domains import Decomposed, ambient, cell_domain
from ..geometry import density
from ..nodes import CT, Index, Indexed, Integral, ISum, Prod, PullBack
from ..static import free, shape, support
from ..substitution import expressions, map_fields


def rule(e):
    # The rows of expand_integrals, for an Integral over a decomposed domain.
    match e:
        case Integral(f, Decomposed() as D) if shape(f):
            K = tuple(Index(("uflx.expand_integrals", f, D, m)) for m in range(len(shape(f))))
            return CT(rule(Integral(Indexed(f, K), D)), K)
        case Integral(f, Decomposed(pieces) as D):
            return total(piece_integral(f, D, j) for j in range(len(pieces)))
    return e


def piece_integral(f, D, j):
    # Pulled back by the cells of D, or by the piece if the support is the ambient domain.
    piece, cells = D.pieces[j], cell_domain(D, j)
    g = PullBack(f, piece if support(f) == ambient(D) else cells)
    term = rule(Integral(Prod(g, density(cells)), piece.P))
    family = tuple(free(piece.psi))
    return ISum(term, family) if family else term


def expand_integrals(e):
    # Inner integrals first.
    return rule(map_fields(e, expand_integrals))


def in_n4(e):
    match e:
        case Integral(_, Decomposed()):
            return False
    return all(in_n4(c) for c in expressions(e))
