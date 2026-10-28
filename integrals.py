from algebra import determinant, gram, tangent_gram, total
from core import CT, Abs, Decomposed, Gradient, Index, Indexed, Integral, ISum, Num, Prod, PullBack, Sqrt, ambient, ambient_dim, cell_domain, map_fields, operands, topdim
from static import free, shape, support


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
    # The integrand is pulled back by the cells of D, or by the map of the piece if its support is the ambient domain.
    piece, cells = D.pieces[j], cell_domain(D, j)
    g = PullBack(f, piece if support(f) == ambient(D) else cells)
    term = rule(Integral(Prod(g, density(cells)), piece.P))
    family = tuple(free(piece.psi))
    return ISum(term, family) if family else term


def density(m):
    k, n = topdim(m.P), ambient_dim(m.Q)
    J = Gradient(m.psi, m.P)
    if k == 0:
        return Num(1)
    if k < ambient_dim(m.P):
        return Sqrt(determinant(tangent_gram(J, m.P), ambient_dim(m.P)))
    if k == n:
        return Abs(determinant(J, k))
    return Sqrt(determinant(gram(J), k))


def expand_integrals(e):
    # Inner integrals first.
    return rule(map_fields(e, expand_integrals))


def in_n4(e):
    match e:
        case Integral(_, Decomposed()):
            return False
    return all(in_n4(c) for c in operands(e))
