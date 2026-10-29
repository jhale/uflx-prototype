from __future__ import annotations

from abc import abstractmethod
from functools import cached_property
from typing import final

from .base import AbstractExpression, Base
from .indices import GatheredIndex, Index
from .topology import AbstractCellType, AbstractTopology


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
    # A cell type that is also a coordinate domain, whose vertices the core does not read.
    pass


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

    @cached_property
    def topology(self):
        # T if the free indices of piece j are (cell_index(T, j),) for every j, and otherwise None.
        found = set()
        for j, piece in enumerate(self.pieces):
            match piece.parameterisation.free_indices:
                case ((Index(("uflx.cell_index", T, jj, 0)), _),) if jj == j:
                    found.add(T)
                case _:
                    return None
        return found.pop() if len(found) == 1 else None


@final
class CellMap(Base):
    identifier = "uflx.CellMap"
    # The characteristic map of the cells of group j of a topology that the entry selects.
    topology: AbstractTopology
    group: int
    entry: Index | GatheredIndex
    opaque = ("group",)

    @property
    def parameter_domain(self):
        return self.topology.cell_types[self.group]

    @property
    def ambient_domain(self):
        return self.topology

    @cached_property
    def free_indices(self):
        if isinstance(self.entry, GatheredIndex):
            return self.entry.free_indices
        return ((self.entry, self.topology.num_cells[self.group]),)


@final
class SideMap(Base):
    identifier = "uflx.SideMap"
    # The lazy map of the cells of the parameter domain into the ambient domain through a side.
    ambient_domain: DecomposedDomain
    parameter_domain: DecomposedDomain
    side: str
    opaque = ("side",)
    free_indices = ()
