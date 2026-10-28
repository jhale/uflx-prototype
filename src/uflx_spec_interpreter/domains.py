from dataclasses import dataclass
from functools import cached_property

from .nodes import Base
from .substitution import substitute


@dataclass(frozen=True, eq=False)
class CoordDomain(Base):
    identifier = "toy.CoordDomain"
    name: str
    dim: int
    opaque = ("name", "dim")

    @property
    def ambient_dim(self):
        return self.dim

    @property
    def topdim(self):
        return self.dim


@dataclass(frozen=True, eq=False)
class DomainMap(Base):
    identifier = "uflx.DomainMap"
    P: Base
    psi: Base
    Q: Base

    @property
    def parameter_domain(self):
        return self.P

    @cached_property
    def free_indices(self):
        return self.psi.free_indices


@dataclass(frozen=True, eq=False)
class DecomposedDomain(Base):
    identifier = "uflx.DecomposedDomain"
    pieces: tuple

    @property
    def ambient_dim(self):
        return self.pieces[0].Q.ambient_dim

    @property
    def topdim(self):
        return self.pieces[0].P.topdim


def ambient_dim(P):
    return P.ambient_dim


def topdim(P):
    return P.topdim


def ambient(P):
    return P.pieces[0].Q if isinstance(P, DecomposedDomain) else P


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
    D: Base
    j: int
    entry: Base
    opaque = ("j",)

    @property
    def P(self):
        return self.D.pieces[self.j].P

    @property
    def Q(self):
        return self.D

    @property
    def parameter_domain(self):
        return self.P

    @cached_property
    def free_indices(self):
        return self.psi.free_indices

    @property
    def psi(self):
        piece = self.D.pieces[self.j]
        family = [i for i, _ in piece.psi.free_indices]
        return (
            substitute(piece.psi, {family[0]: self.entry})
            if family and family[0] != self.entry
            else piece.psi
        )


@dataclass(frozen=True, eq=False)
class SideMap(Base):
    identifier = "uflx.SideMap"
    # The map from the cells of E into D, from one side, through the side tables.
    D: Base
    E: Base
    cells: Base
    positions: Base
    free_indices = ()

    @property
    def parameter_domain(self):
        return self.E


def cell_domain(D, j):
    family = [i for i, _ in D.pieces[j].psi.free_indices]
    return CellMap(D, j, family[0]) if family else DomainMap(D.pieces[j].P, D.pieces[j].psi, D)
