from .nodes import CellMap, CoordinateDomain, DecomposedDomain, DomainMap, Normal, SideMap


def cell_domain(D, j):
    # Piece j of D, as one family of cells when the piece has a cell index.
    piece = D.pieces[j]
    family = [i for i, _ in piece.parameterisation.free_indices]
    if family:
        return CellMap(D, j, family[0])
    return DomainMap(piece.parameter_domain, piece.parameterisation, D)


def coordinate_domain(name, dimension):
    return CoordinateDomain(name, dimension)


def decomposed_domain(pieces):
    return DecomposedDomain(tuple(pieces))


def gathered_cell_domain(D, j, cell_index):
    return CellMap(D, j, cell_index)


def side_map(ambient_domain, parameter_domain, cells, positions):
    return SideMap(ambient_domain, parameter_domain, cells, positions)


def normal(ambient_domain, parameter_domain, side):
    return Normal(ambient_domain, parameter_domain, side)
