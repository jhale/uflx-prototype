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
from .tensors import component
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
    chi = component(ComponentTensor(Indexed(piece.parameterisation, (k,)), (c, k)), (g,))
    return DomainMap(piece.parameter_domain, chi, D)


def cell_of(m):
    # The type i of the cells if m is cell_domain(E, i), and otherwise None.
    E = m.ambient_domain
    if isinstance(m, DomainMap) and isinstance(E, DecomposedDomain):
        return next((i for i in range(len(E.pieces)) if m == cell_domain(E, i)), None)
    return None


def is_cell(m):
    # Whether m is a cell domain or a gathered cell domain.
    if cell_of(m) is not None:
        return True
    match m:
        case DomainMap(
            _, ComponentTensor(Indexed(ComponentTensor(), (GatheredIndex() as g, _)), _)
        ):
            E = m.ambient_domain
            pieces = range(len(E.pieces)) if isinstance(E, DecomposedDomain) else ()
            return any(m == gathered_cell_domain(E, j, g) for j in pieces)
    return False


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
    return gathered_cell_domain(D, j, GatheredIndex(cells, (cell_index(S, i),))), A


def normal(sm):
    return Normal(sm)
