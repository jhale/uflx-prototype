from dataclasses import dataclass

from .nodes import Node
from .substitution import substitute


@dataclass(frozen=True, eq=False)
class CoordDomain(Node):
    identifier = "toy.CoordDomain"
    name: str
    dim: int
    opaque = ("name", "dim")


@dataclass(frozen=True, eq=False)
class DomainMap(Node):
    identifier = "uflx.DomainMap"
    P: Node
    psi: Node
    Q: Node


@dataclass(frozen=True, eq=False)
class Decomposed(Node):
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
class Gathered(Node):
    identifier = "uflx.GatheredIndex"
    # An index read from a table at the given entries.
    table: Node
    indices: tuple


@dataclass(frozen=True, eq=False)
class CellMap(Node):
    identifier = "toy.CellMap"
    # Piece j of D seen as a domain in D, for the cell selected by entry (an Index or a Gathered).
    D: Node
    j: int
    entry: Node
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
        # static imports this module, so the import is local.
        from .static import free

        family = list(free(piece.psi))
        return (
            substitute(piece.psi, {family[0]: self.entry})
            if family and family[0] != self.entry
            else piece.psi
        )


@dataclass(frozen=True, eq=False)
class SideMap(Node):
    identifier = "uflx.SideMap"
    # The map from the cells of E into D, from one side, through the side tables.
    D: Node
    E: Node
    cells: Node
    positions: Node


def cell_domain(D, j):
    from .static import free

    family = list(free(D.pieces[j].psi))
    return CellMap(D, j, family[0]) if family else DomainMap(D.pieces[j].P, D.pieces[j].psi, D)
