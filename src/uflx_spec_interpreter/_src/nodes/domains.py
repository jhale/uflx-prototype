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
    parameter_domain: AbstractCoordinateDomain | AbstractTopology
    parameterisation: AbstractExpression
    ambient_domain: AbstractDomain

    @cached_property
    def free_indices(self):
        return self.parameterisation.free_indices


@final
class Mesh(AbstractDomain):
    identifier = "uflx.Mesh"
    coordinate_function: AbstractExpression
    ambient_domain: AbstractDomain

    def __init__(self, coordinate_function, ambient_domain):
        if not isinstance(coordinate_function, AbstractExpression):
            raise TypeError("mesh coordinates must be an expression")
        if not isinstance(ambient_domain, AbstractDomain):
            raise TypeError("a mesh must have an ambient domain")
        if not isinstance(coordinate_function.support, AbstractTopology):
            raise TypeError("mesh coordinates must be supported on a topology")
        if coordinate_function.free_indices:
            raise ValueError("mesh coordinates must have no free indices")

    @property
    def topology(self):
        return self.coordinate_function.support

    @property
    def dimension(self):
        return self.topology.dimension

    @cached_property
    def pieces(self):
        # Derived views for integration and the oracle, not construction arguments.
        from ..domains import cell_map
        from .calculus import PullBack

        return tuple(
            DomainMap(
                K,
                PullBack(self.coordinate_function, cell_map(self.topology, j)),
                self.ambient_domain,
            )
            for j, K in enumerate(self.topology.cell_types)
        )


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
    ambient_domain: Mesh | AbstractTopology
    parameter_domain: Mesh | AbstractTopology
    sides: tuple
    opaque = ("sides",)
    free_indices = ()

    @property
    def side_topology(self):
        P = self.parameter_domain
        return P if isinstance(P, AbstractTopology) else P.topology
