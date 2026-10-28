from .nodes import CellMap, DecomposedDomain, DomainMap, Normal, SideMap


def cell_domain(D, j):
    # Piece j of D, as one family of cells when the piece has a cell index.
    piece = D.pieces[j]
    family = [i for i, _ in piece.parameterisation.free_indices]
    if family:
        return CellMap(D, j, family[0])
    return DomainMap(piece.parameter_domain, piece.parameterisation, D)


def decomposed_domain(pieces):
    return DecomposedDomain(tuple(pieces))


def gathered_cell_domain(D, j, g):
    return CellMap(D, j, g)


def side_map(ambient_domain, parameter_domain, cells, positions):
    return SideMap(ambient_domain, parameter_domain, cells, positions)


def normal(D, E, s):
    return Normal(D, E, s)
