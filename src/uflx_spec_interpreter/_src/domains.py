from .nodes import (
    ComponentTensor,
    DecomposedDomain,
    DomainMap,
    Index,
    Indexed,
    Normal,
    SideMap,
)


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


def side_map(ambient_domain, parameter_domain, cells, positions):
    return SideMap(ambient_domain, parameter_domain, cells, positions)


def normal(D, E, s):
    return Normal(D, E, s)
