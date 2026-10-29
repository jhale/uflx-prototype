from functools import cache

from .nodes import (
    CellMap,
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
from .substitution import everywhere, substitute
from .tensors import component
from .topology import cell_index


def cell_domain(D, j, e=None):
    # Piece j with the ambient domain D, and with e in place of the cell index for cell e.
    piece = D.pieces[j]
    psi = piece.parameterisation
    if e is not None:
        psi = substitute(psi, {cell_index(D.topology, j): e})
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
    # The expression on a topology that a mesh is built from, or None for another domain.
    found = set()
    for j, piece in enumerate(getattr(D, "pieces", ())):
        match piece.parameterisation:
            case PullBack(x, CellMap(T, jj, e)) if jj == j and e == cell_index(T, j):
                found.add(x)
            case _:
                return None
    return found.pop() if len(found) == 1 else None


@cache
def mesh_cell(m):
    # The group and the entry of a cell of a mesh, read from the cell maps of the
    # parameterisation. A pass rewrites the mesh and the cell alike, so the comparison holds.
    if not isinstance(m, DomainMap) or not isinstance(m.ambient_domain, DecomposedDomain):
        return None
    D = m.ambient_domain
    found = {(c.group, c.entry) for c in cell_maps(m.parameterisation) if c.topology == D.topology}
    if len(found) == 1:
        j, e = found.pop()
        if m == cell_domain(D, j, e):
            return j, e
    return None


def cell_maps(e):
    found = []

    def visit(n):
        if isinstance(n, PullBack) and isinstance(n.domain, CellMap):
            found.append(n.domain)
        return True

    everywhere(e, visit)
    return found


def decomposed_domain(pieces):
    return DecomposedDomain(tuple(pieces))


def gathered_cell_domain(D, j, g):
    # The cells of piece j that g selects, through the one free index c of the piece.
    if coordinate_function(D) is not None:
        return cell_domain(D, j, g)
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
    if cell_of(m) is not None or mesh_cell(m) is not None:
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
