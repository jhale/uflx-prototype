from dataclasses import dataclass
from functools import cached_property

from .nodes import Base
from .substitution import substitute


@dataclass(frozen=True, eq=False)
class CoordDomain(Base):
    identifier = "toy.CoordDomain"
    name: str
    dimension: int
    opaque = ("name", "dimension")

    @property
    def ambient_dim(self):
        return self.dimension

    @property
    def ambient_domain(self):
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
    def ambient_dim(self):
        return self.pieces[0].ambient_domain.ambient_dim

    @property
    def ambient_domain(self):
        return self.pieces[0].ambient_domain

    @property
    def dimension(self):
        return self.pieces[0].parameter_domain.dimension


def ambient_dim(P):
    return P.ambient_dim


@dataclass(frozen=True, eq=False)
class GatheredIndex(Base):
    identifier = "uflx.GatheredIndex"
    # An index read from a table at the given entries.
    table: Base
    indices: tuple

    @cached_property
    def free_indices(self):
        # Each Index entry takes the extent of its axis of the table.
        shape = self.table.value_shape
        return tuple((j, shape[m]) for m, j in enumerate(self.indices) if not isinstance(j, int))


@dataclass(frozen=True, eq=False)
class CellMap(Base):
    identifier = "toy.CellMap"
    # Piece j of D seen as a domain in D, for the cell selected by entry (an Index or a Gathered).
    ambient_domain: Base
    j: int
    entry: Base
    opaque = ("j",)

    @property
    def parameter_domain(self):
        return self.ambient_domain.pieces[self.j].parameter_domain

    @cached_property
    def free_indices(self):
        return self.parameterisation.free_indices

    @property
    def parameterisation(self):
        piece = self.ambient_domain.pieces[self.j]
        family = [i for i, _ in piece.parameterisation.free_indices]
        return (
            substitute(piece.parameterisation, {family[0]: self.entry})
            if family and family[0] != self.entry
            else piece.parameterisation
        )


@dataclass(frozen=True, eq=False)
class SideMap(Base):
    identifier = "uflx.SideMap"
    # The map from the cells of E into D, from one side, through the side tables.
    ambient_domain: Base
    parameter_domain: Base
    cells: Base
    positions: Base
    free_indices = ()


def cell_domain(D, j):
    piece = D.pieces[j]
    family = [i for i, _ in piece.parameterisation.free_indices]
    return (
        CellMap(D, j, family[0])
        if family
        else DomainMap(piece.parameter_domain, piece.parameterisation, D)
    )
