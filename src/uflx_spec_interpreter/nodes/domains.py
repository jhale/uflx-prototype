from dataclasses import dataclass
from functools import cached_property

from .base import Base


@dataclass(frozen=True, eq=False)
class CoordinateDomain(Base):
    identifier = "toy.CoordinateDomain"
    # A coordinate domain, which stands for both RD and a reference cell.
    name: str
    dimension: int
    opaque = ("name", "dimension")

    @property
    def ambient_domain(self):
        return self

    @property
    def ambient_coordinate_domain(self):
        return self


@dataclass(frozen=True, eq=False)
class DomainMap(Base):
    identifier = "uflx.DomainMap"
    parameter_domain: Base
    parameterisation: Base
    ambient_domain: Base

    @cached_property
    def free_indices(self):
        return self.parameterisation.free_indices


@dataclass(frozen=True, eq=False)
class DecomposedDomain(Base):
    identifier = "uflx.DecomposedDomain"
    pieces: tuple

    @property
    def ambient_domain(self):
        return self.pieces[0].ambient_domain

    @property
    def ambient_coordinate_domain(self):
        return self.ambient_domain.ambient_coordinate_domain

    @property
    def dimension(self):
        return self.pieces[0].parameter_domain.dimension


@dataclass(frozen=True, eq=False)
class CellMap(Base):
    identifier = "toy.CellMap"
    # The cells of one piece, selected by the cell index, an Index or a GatheredIndex.
    ambient_domain: Base
    piece: int
    cell_index: Base
    opaque = ("piece",)

    @property
    def parameter_domain(self):
        return self.ambient_domain.pieces[self.piece].parameter_domain

    @cached_property
    def free_indices(self):
        return self.parameterisation.free_indices

    @property
    def parameterisation(self):
        # The substitution imports the nodes, so the import is local.
        from ..substitution import substitute

        piece = self.ambient_domain.pieces[self.piece]
        family = [i for i, _ in piece.parameterisation.free_indices]
        if family and family[0] != self.cell_index:
            return substitute(piece.parameterisation, {family[0]: self.cell_index})
        return piece.parameterisation


@dataclass(frozen=True, eq=False)
class SideMap(Base):
    identifier = "uflx.SideMap"
    # The cells of the parameter domain in the ambient domain from one side, by the side tables.
    ambient_domain: Base
    parameter_domain: Base
    cells: Base
    positions: Base
    free_indices = ()


def cell_domain(D, j):
    # Piece j of D, as one family of cells when the piece has a cell index.
    piece = D.pieces[j]
    family = [i for i, _ in piece.parameterisation.free_indices]
    if family:
        return CellMap(D, j, family[0])
    return DomainMap(piece.parameter_domain, piece.parameterisation, D)
