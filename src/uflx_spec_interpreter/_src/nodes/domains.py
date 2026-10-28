from __future__ import annotations

from abc import abstractmethod
from functools import cached_property
from typing import TYPE_CHECKING, final

from .base import AbstractExpression, Base
from .indices import Index
from .topology import AbstractCellType

if TYPE_CHECKING:
    from .indices import AbstractIndexValue, GatheredIndex


class AbstractDomain(Base):
    @property
    @abstractmethod
    def dimension(self) -> int: ...

    @property
    @abstractmethod
    def ambient_domain(self) -> AbstractDomain: ...

    @property
    def ambient_coordinate_domain(self) -> AbstractCoordinateDomain:
        # The first coordinate domain among the ambient domains.
        return self.ambient_domain.ambient_coordinate_domain


class AbstractCoordinateDomain(AbstractDomain):
    @property
    def ambient_domain(self):
        return self

    @property
    def ambient_coordinate_domain(self):
        return self


@final
class RD(AbstractCoordinateDomain):
    identifier = "uflx.RD"
    dimension: int
    opaque = ("dimension",)


class AbstractReferenceCell(AbstractCoordinateDomain, AbstractCellType):
    # A cell type whose vertices have coordinates, row v of value shape (m, n) for vertex v.
    @property
    @abstractmethod
    def vertices(self) -> AbstractExpression: ...


@final
class DomainMap(Base):
    identifier = "uflx.DomainMap"
    parameter_domain: AbstractDomain
    parameterisation: AbstractExpression
    ambient_domain: AbstractDomain

    @cached_property
    def free_indices(self):
        return self.parameterisation.free_indices


@final
class DecomposedDomain(AbstractDomain):
    identifier = "uflx.DecomposedDomain"
    pieces: tuple[DomainMap, ...]

    @property
    def ambient_domain(self):
        return self.pieces[0].ambient_domain

    @property
    def dimension(self):
        return self.pieces[0].parameter_domain.dimension


@final
class CellMap(Base):
    identifier = "toy.CellMap"
    # The cells of one piece, selected by the cell index, an Index or a GatheredIndex.
    ambient_domain: DecomposedDomain
    piece: int
    cell_index: Index | GatheredIndex
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


@final
class SideMap(Base):
    identifier = "uflx.SideMap"
    # The cells of the parameter domain in the ambient domain from one side, by the side tables.
    ambient_domain: DecomposedDomain
    parameter_domain: DecomposedDomain
    cells: AbstractIndexValue
    positions: AbstractExpression
    free_indices = ()
