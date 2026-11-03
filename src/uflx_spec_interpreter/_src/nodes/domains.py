from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, cast, final

from ..interning import cached
from .base import AbstractExpression, Base
from .indices import GatheredIndex, Index
from .scalar_types import ScalarType
from .topology import AbstractCellType, AbstractTopology

if TYPE_CHECKING:
    from ..nodes.indices import AbstractExtent
    from .domains import AbstractCoordinateDomain, AbstractDomain, DomainMap


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
    def ambient_domain(self) -> AbstractDomain:
        return self

    @property
    def ambient_coordinate_domain(self) -> AbstractCoordinateDomain:
        return self


@final
class RD(AbstractCoordinateDomain):
    identifier = "uflx.RD"
    dimension: int  # pyright: ignore[reportIncompatibleMethodOverride]

    def _validate(self) -> None:
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

    @property
    @cached
    def free_indices(self) -> tuple[tuple[Index, int | AbstractExtent], ...]:
        return self.parameterisation.free_indices

    def _validate(self) -> None:
        from ..validation import require

        f = self.parameterisation
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
    ambient_domain: AbstractCoordinateDomain  # pyright: ignore[reportIncompatibleMethodOverride]

    @property
    def topology(self) -> AbstractTopology:
        domain = self.coordinate_function.domain
        if not isinstance(domain, AbstractTopology):
            raise TypeError("mesh coordinates require a topology")
        return domain

    @property
    def dimension(self) -> int:
        return self.topology.dimension

    @property
    @cached
    def pieces(self) -> tuple[DomainMap, ...]:
        # Derived views for integration and the oracle, not construction arguments.
        from ..domains import cell_map
        from .calculus import PullBack
        from .domains import DomainMap

        cells = self.topology.cell_types
        if not all(isinstance(cell, AbstractReferenceCell) for cell in cells):
            raise TypeError("mesh cells must be reference cells")
        return tuple(
            DomainMap(
                cast("AbstractReferenceCell", K),
                PullBack(self.coordinate_function, cell_map(self.topology, j)),
                self.ambient_domain,
            )
            for j, K in enumerate(self.topology.cell_types)
        )

    def _validate(self) -> None:
        from ..validation import require

        f = self.coordinate_function
        topology = self.topology
        require(
            all(isinstance(K, AbstractReferenceCell) for K in topology.cell_types),
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
            topology.dimension <= self.ambient_domain.dimension,
            "mesh dimension exceeds ambient dimension",
        )


@final
class CellMap(Base):
    identifier = "uflx.CellMap"
    ambient_domain: AbstractTopology | Mesh
    cell_group: int
    entry: int | Index | GatheredIndex

    @property
    def topology(self) -> AbstractTopology:
        X = self.ambient_domain
        return X.topology if isinstance(X, Mesh) else X

    @property
    def parameter_domain(self) -> AbstractReferenceCell:
        cell = self.topology.cell_types[self.cell_group]
        if not isinstance(cell, AbstractReferenceCell):
            raise TypeError("cell must be a reference cell")
        return cell

    @property
    @cached
    def parameterisation(self) -> AbstractExpression:
        from .calculus import PullBack

        if not isinstance(self.ambient_domain, Mesh):
            raise TypeError("a topology cell has no parameterisation")
        return PullBack(
            self.ambient_domain.coordinate_function,
            CellMap(self.topology, self.cell_group, self.entry),
        )

    @property
    @cached
    def free_indices(self) -> tuple[tuple[Index, int | AbstractExtent], ...]:
        if isinstance(self.entry, GatheredIndex):
            return self.entry.free_indices
        if isinstance(self.entry, int):
            return ()
        return ((self.entry, self.topology.num_cells[self.cell_group]),)

    def _validate(self) -> None:
        from ..validation import require

        if type(self.cell_group) is not int:
            raise TypeError("cell group must be an integer")
        require(0 <= self.cell_group < len(self.topology.cell_types), "invalid cell group")
        self.parameter_domain  # Check that the selected cell is a reference cell.
        size = self.topology.num_cells[self.cell_group]
        if isinstance(self.entry, int):
            require(
                type(size) is int and 0 <= self.entry < size,
                "cell selection is outside the group",
            )
        elif isinstance(self.entry, GatheredIndex):
            require(self.entry.extent == size, "cell gather has the wrong extent")
        else:
            pass


@final
class SideMap(Base):
    identifier = "uflx.SideMap"
    ambient_domain: Mesh | AbstractTopology
    parameter_domain: Mesh | AbstractTopology
    side: str
    free_indices = ()

    @property
    def side_topology(self) -> AbstractTopology:
        P = self.parameter_domain
        return P if isinstance(P, AbstractTopology) else P.topology

    def _validate(self) -> None:
        from ..validation import require

        X = self.ambient_domain
        S = self.side_topology
        T = X.topology if isinstance(X, Mesh) else X
        require(
            all(isinstance(K, AbstractReferenceCell) for K in S.cell_types),
            "side cells must be reference cells",
        )
        require(
            len(S.sides) == len(S.cell_types)
            and all(
                any((name == self.side and U == T for name, U, _ in sides)) for sides in S.sides
            ),
            "side is absent from a cell group",
        )
