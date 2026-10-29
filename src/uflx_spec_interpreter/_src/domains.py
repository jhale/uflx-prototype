from .nodes import (
    CellMap,
    Coordinate,
    DecomposedDomain,
    DomainMap,
    GatheredIndex,
    Normal,
    PullBack,
    SideMap,
)
from .topology import cell_index


def cell_domain(D, j, e=None):
    # Piece j with the ambient domain D, and for a mesh cell e of group j through its cell map.
    piece = D.pieces[j]
    psi = piece.parameterisation
    if e is not None:
        x, c = psi.expression, psi.domain
        psi = PullBack(x, CellMap(c.topology, c.group, e))
    return DomainMap(piece.parameter_domain, psi, D)


def cell_domain_of_specification(D, j):
    # The signature of LANGUAGE.md, which the draft of coordinate functions extends with e.
    return cell_domain(D, j)


def cell_map(T, j, e=None):
    return CellMap(T, j, cell_index(T, j) if e is None else e)


def mesh(x, Q):
    T = x.support
    pieces = (DomainMap(K, PullBack(x, cell_map(T, j)), Q) for j, K in enumerate(T.cell_types))
    return DecomposedDomain(tuple(pieces))


def coordinate_function(D):
    # The expression on a topology that a mesh is built from.
    return D.pieces[0].parameterisation.expression


def decomposed_domain(pieces):
    return DecomposedDomain(tuple(pieces))


def gathered_cell_domain(D, j, g):
    return cell_domain(D, j, g)


def cell_of(m):
    # The group and the entry of a cell: of a mesh by its cell map, and of a decomposed domain
    # with explicit pieces, such as a side domain, by its piece.
    match m:
        case DomainMap(_, PullBack(_, CellMap(_, j, e)), D) if m == cell_domain(D, j, e):
            return j, e
        case DomainMap(_, psi, DecomposedDomain(pieces) as D):
            for j in range(len(pieces)):
                if m == cell_domain(D, j):
                    # A piece with no free index is the one member of its family.
                    return j, next((i for i, _ in psi.free_indices), None)
    return None


def is_cell(m):
    return cell_of(m) is not None


def side_map(D, E, s):
    return SideMap(D, E, s)


def side_domain(D, S):
    # Piece i is the coordinates of D through the first side of group i on the topology of D.
    pieces = []
    for i, F in enumerate(S.cell_types):
        s = next(name for name, T, _ in S.sides[i] if T == D.topology)
        B, A = inclusion(D, S, i, s)
        pieces.append(DomainMap(F, PullBack(PullBack(Coordinate(D), B), A), D.ambient_domain))
    return DecomposedDomain(tuple(pieces))


def inclusion(D, S, i, s):
    # B, the cell on the side s of each cell e of group i, and A, the inclusion of e into it.
    (_, _, j) = next(side for side in S.sides[i] if side[0] == s)
    cells, A, _ = S.side_tables(i, s)
    return cell_domain(D, j, GatheredIndex(cells, (cell_index(S, i),))), A


def normal(sm):
    return Normal(sm)
