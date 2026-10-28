from ..algebra import total
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
    cell_domain,
)
from ..static import free, shape, support
from ..substitution import expressions, map_fields


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


def expand_integrals(e):
    # Inner integrals first.
    return rule(map_fields(e, expand_integrals))


def in_n4(e):
    match e:
        case Integral(_, DecomposedDomain()):
            return False
    return all(in_n4(c) for c in expressions(e))
