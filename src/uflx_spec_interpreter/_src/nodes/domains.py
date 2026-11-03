from __future__ import annotations

from abc import abstractmethod
from functools import cached_property
from typing import final

from .base import AbstractExpression, Base
from .indices import GatheredIndex, Index
from .scalar_types import ScalarType
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

    def _validate(self):
        from ..validation import require

        if type(self.dimension) is not int:
            raise TypeError("dimension must be an integer")
        require(self.dimension >= 0, "dimension must be nonnegative")


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

    def _validate(self):
        from ..validation import require, typed

        typed(
            self.parameter_domain,
            (AbstractCoordinateDomain, AbstractTopology),
            "invalid parameter domain",
        )
        typed(
            self.ambient_domain, AbstractCoordinateDomain, "map target must be a coordinate domain"
        )
        f = self.parameterisation
        typed(f, AbstractExpression, "parameterisation must be an expression")
        require(f.domain in (None, self.parameter_domain), "parameterisation domain differs")
        require(
            f.scalar_type == ScalarType.real
            and (not f.argument_spaces)
            and (f.value_shape == (self.ambient_domain.dimension,)),
            "invalid coordinate expression",
        )
        require(
            self.parameter_domain.dimension <= self.ambient_domain.dimension,
            "map decreases ambient dimension",
        )


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

    def _validate(self):
        from ..validation import require, typed

        typed(
            self.ambient_domain, AbstractCoordinateDomain, "mesh target must be a coordinate domain"
        )
        f = self.coordinate_function
        typed(f, AbstractExpression, "coordinates must be an expression")
        typed(f.domain, AbstractTopology, "coordinates must have a topology domain")
        require(
            all(isinstance(K, AbstractReferenceCell) for K in f.domain.cell_types),
            "mesh cells must be reference cells",
        )
        require(not f.free_indices, "mesh coordinates must have no free indices")
        require(
            not f.free_indices
            and (not f.argument_spaces)
            and (f.scalar_type == ScalarType.real)
            and (f.value_shape == (self.ambient_domain.dimension,)),
            "invalid mesh coordinates",
        )
        require(
            f.domain.dimension <= self.ambient_domain.dimension,
            "mesh dimension exceeds ambient dimension",
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

    def _validate(self):
        from ..validation import require, typed

        typed(self.ambient_domain, (AbstractTopology, Mesh), "cell map requires a topology or mesh")
        if type(self.cell_group) is not int:
            raise TypeError("cell group must be an integer")
        require(0 <= self.cell_group < len(self.topology.cell_types), "invalid cell group")
        require(
            isinstance(self.parameter_domain, AbstractReferenceCell),
            "cell must be a reference cell",
        )
        size = self.topology.num_cells[self.cell_group]
        if type(self.entry) is int:
            require(
                type(size) is int and 0 <= self.entry < size,
                "cell selection is outside the group",
            )
        elif isinstance(self.entry, GatheredIndex):
            require(self.entry.extent == size, "cell gather has the wrong extent")
        else:
            typed(self.entry, Index, "invalid cell entry")


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

    def _validate(self):
        from ..validation import require, typed

        typed(self.side, str, "side must be a name")
        typed(self.ambient_domain, (AbstractTopology, Mesh), "invalid side target")
        P, X = (self.parameter_domain, self.ambient_domain)
        typed(
            P,
            Mesh if isinstance(X, Mesh) else AbstractTopology,
            "side source and target kinds differ",
        )
        S = self.side_topology
        T = X.topology if isinstance(X, Mesh) else X
        require(
            all(isinstance(K, AbstractReferenceCell) for K in S.cell_types),
            "side cells must be reference cells",
        )
        require(
            len(S.sides) == len(S.cell_types)
            and all(
                any((name == self.side and U == T for name, U, j in sides)) for sides in S.sides
            ),
            "side is absent from a cell group",
        )
