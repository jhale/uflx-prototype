from dataclasses import dataclass

from .nodes import Base
from .substitution import substitute


@dataclass(frozen=True, eq=False)
class CoordDomain(Base):
    identifier = "toy.CoordDomain"
    name: str
    dim: int
    opaque = ("name", "dim")


@dataclass(frozen=True, eq=False)
class DomainMap(Base):
    identifier = "uflx.DomainMap"
    P: Base
    psi: Base
    Q: Base


@dataclass(frozen=True, eq=False)
class Decomposed(Base):
    identifier = "uflx.DecomposedDomain"
    pieces: tuple


def ambient_dim(P):
    if isinstance(P, CoordDomain):
        return P.dim
    return ambient_dim(P.pieces[0].Q)


def topdim(P):
    return P.dim if isinstance(P, CoordDomain) else topdim(P.pieces[0].P)


def ambient(P):
    return P.pieces[0].Q if isinstance(P, Decomposed) else P


@dataclass(frozen=True, eq=False)
class Gathered(Base):
    identifier = "uflx.GatheredIndex"
    # An index read from a table at the given entries.
    table: Base
    indices: tuple


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


def cell_domain(D, j):
    family = [i for i, _ in D.pieces[j].psi.free_indices]
    return CellMap(D, j, family[0]) if family else DomainMap(D.pieces[j].P, D.pieces[j].psi, D)
