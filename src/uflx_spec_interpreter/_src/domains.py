from .cells import affine_map
from .nodes import (
    ComponentTensor,
    Coordinate,
    DecomposedDomain,
    DomainMap,
    GatheredIndex,
    Index,
    Indexed,
    Normal,
    PullBack,
    SideMap,
)
from .topology import cell_index


def cell_domain(D, j):
    piece = D.pieces[j]
    return DomainMap(piece.parameter_domain, piece.parameterisation, D)


def decomposed_domain(pieces):
    return DecomposedDomain(tuple(pieces))


def gathered_cell_domain(D, j, g):
    # The cells of piece j that g selects, through the one free index c of the piece.
    piece = D.pieces[j]
    ((c, _),) = piece.parameterisation.free_indices
    k = Index(("uflx.gathered_cell_domain", D, j, g, 0))
    by_cell = ComponentTensor(Indexed(piece.parameterisation, (k,)), (c, k))
    chi = ComponentTensor(Indexed(by_cell, (g, k)), (k,))
    return DomainMap(piece.parameter_domain, chi, D)


def cell_of(m):
    # The type i of the cells if m is cell_domain(E, i), and otherwise None.
    E = m.ambient_domain
    if isinstance(m, DomainMap) and isinstance(E, DecomposedDomain):
        return next((i for i in range(len(E.pieces)) if m == cell_domain(E, i)), None)
    return None


def side_map(D, E, s):
    return SideMap(D, E, s)


def restriction(f, E, s):
    return f if f.support is None else PullBack(f, SideMap(f.support, E, s))


def side_domain(D, S):
    # Piece i is the coordinates of D through the inclusion of group i from its first side.
    pieces = []
    for i, F in enumerate(S.cell_types):
        B, A = inclusion(D, S, i, S.sides[i][0][0])
        pieces.append(DomainMap(F, PullBack(PullBack(Coordinate(D), B), A), D.ambient_domain))
    return DecomposedDomain(tuple(pieces))


def inclusion(D, S, i, s):
    # B, the cell on the side s of each cell e of group i, and A, the inclusion of e into it.
    (_, _, j) = next(side for side in S.sides[i] if side[0] == s)
    cells, _ = S.side_tables(i, s)
    B = gathered_cell_domain(D, j, GatheredIndex(cells, (cell_index(S, i),)))
    F, K_j = S.cell_types[i], D.pieces[j].parameter_domain
    return B, DomainMap(F, affine_map(F, inclusion_points(S, i, s)), K_j)


def inclusion_points(S, i, s):
    # The position in K_j of each vertex v of each cell e of group i, from the side s.
    _, positions = S.side_tables(i, s)
    e = cell_index(S, i)
    v, k = (Index(("uflx.inclusion", S, i, s, n)) for n in range(2))
    return ComponentTensor(Indexed(positions, (e, v, k)), (v, k))


def normal(D, E, s):
    return Normal(D, E, s)
