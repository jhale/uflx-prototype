from ..arithmetic import total
from ..domains import cell_domain
from ..geometry import density
from ..nodes import (
    ComponentTensor,
    DecomposedDomain,
    Index,
    Indexed,
    IndexSum,
    Integral,
    Product,
    PullBack,
)
from ..static import free, shape, support
from ..substitution import bottom_up, everywhere


def rule(e):
    # The rows of expand_integrals, for an Integral over a decomposed domain.
    match e:
        case Integral(f, DecomposedDomain() as D) if shape(f):
            K = tuple(Index(("uflx.expand_integrals", f, D, m)) for m in range(len(shape(f))))
            return ComponentTensor(rule(Integral(Indexed(f, K), D)), K)
        case Integral(f, DecomposedDomain(pieces) as D):
            return total(piece_integral(f, D, j) for j in range(len(pieces)))
    return e


def piece_integral(f, D, j):
    # Pulled back by the cells of D, or by the piece if the support is the ambient domain.
    piece, cells = D.pieces[j], cell_domain(D, j)
    g = PullBack(f, piece if support(f) == D.ambient_domain else cells)
    term = rule(Integral(Product(g, density(cells)), piece.parameter_domain))
    family = tuple(free(piece.parameterisation))
    return IndexSum(term, family) if family else term


def expand_integrals(f):
    # Inner integrals first.
    return bottom_up(f, rule)


def in_n4(e):
    def holds(n):
        match n:
            case Integral(_, DecomposedDomain()):
                return False
        return True

    return everywhere(e, holds)
