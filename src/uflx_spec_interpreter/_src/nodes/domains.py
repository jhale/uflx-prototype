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
    ambient_domain: AbstractCoordinateDomain

    @cached_property
    def free_indices(self):
        return self.parameterisation.free_indices


@final
class Mesh(AbstractDomain):
    identifier = "uflx.Mesh"
    coordinate_function: AbstractExpression
    ambient_domain: AbstractCoordinateDomain

    @property
    def topology(self):
        return self.coordinate_function.domain

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
    ambient_domain: AbstractTopology | Mesh
    cell_group: int
    entry: int | Index | GatheredIndex

    @property
    def topology(self):
        X = self.ambient_domain
        return X.topology if isinstance(X, Mesh) else X

    @property
    def parameter_domain(self):
        return self.topology.cell_types[self.cell_group]

    @cached_property
    def parameterisation(self):
        from .calculus import PullBack

        if not isinstance(self.ambient_domain, Mesh):
            raise TypeError("a topology cell has no parameterisation")
        return PullBack(
            self.ambient_domain.coordinate_function,
            CellMap(self.topology, self.cell_group, self.entry),
        )

    @cached_property
    def free_indices(self):
        if isinstance(self.entry, GatheredIndex):
            return self.entry.free_indices
        if type(self.entry) is int:
            return ()
        return ((self.entry, self.topology.num_cells[self.cell_group]),)


@final
class SideMap(Base):
    identifier = "uflx.SideMap"
    ambient_domain: Mesh | AbstractTopology
    parameter_domain: Mesh | AbstractTopology
    side: str
    free_indices = ()

    @property
    def side_topology(self):
        P = self.parameter_domain
        return P if isinstance(P, AbstractTopology) else P.topology
