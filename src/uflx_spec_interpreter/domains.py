from .nodes import CellMap, DomainMap


def cell_domain(D, j):
    # Piece j of D, as one family of cells when the piece has a cell index.
    piece = D.pieces[j]
    family = [i for i, _ in piece.parameterisation.free_indices]
    if family:
        return CellMap(D, j, family[0])
    return DomainMap(piece.parameter_domain, piece.parameterisation, D)
