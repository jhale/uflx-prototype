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
        # The topology of a mesh, the domain built from its coordinate function, and otherwise None.
        from ..domains import mesh
        from .calculus import PullBack

        psi = self.pieces[0].parameterisation
        if not (isinstance(psi, PullBack) and isinstance(psi.domain, CellMap)):
            return None
        return psi.domain.topology if self == mesh(psi.expression, self.ambient_domain) else None


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
    # The restriction from a topology or a mesh to the cells of a topology with sides, through a
    # side of each group: to the side topology for a topology, and to a side domain for a mesh.
    ambient_domain: DecomposedDomain | AbstractTopology
    parameter_domain: DecomposedDomain | AbstractTopology
    sides: tuple
    opaque = ("sides",)
    free_indices = ()

    @property
    def side_topology(self):
        P = self.parameter_domain
        return P if isinstance(P, AbstractTopology) else P.topology
