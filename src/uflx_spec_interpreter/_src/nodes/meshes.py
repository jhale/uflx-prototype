"""Meshes, their cells and sides, and mapped fields."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, cast, final

from .base import AbstractExpression, Base
from .domains import AbstractCoordinateDomain, AbstractDomain, AbstractReferenceCell
from .indices import GatheredIndex, Index
from .interning import _cached
from .scalar_types import ScalarType
from .topology import AbstractTopology

if TYPE_CHECKING:
    from .indices import AbstractExtent
    from .maps import DomainMap
    from .spaces import AbstractFunctionSpace


@final
class Mesh(AbstractDomain):
    identifier = "uflx.Mesh"
    __match_args__ = ("coordinate_function", "ambient_domain")

    def __init__(
        self, coordinate_function: AbstractExpression, ambient_domain: AbstractCoordinateDomain, /
    ) -> None:
        # Base.__new__ stores and validates the construction arguments.
        pass

    coordinate_function: AbstractExpression

    @property
    def ambient_domain(self) -> AbstractCoordinateDomain:
        return cast("AbstractCoordinateDomain", self.construction_arguments[1])

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
    @_cached
    def pieces(self) -> tuple[DomainMap, ...]:
        # Derived views for integration and the oracle, not construction arguments.
        from ..meshes import cell_map
        from .maps import DomainMap, PullBack

        cells = self.topology.cell_types
        if not all(isinstance(cell, AbstractReferenceCell) for cell in cells):
            raise TypeError("mesh cells must be reference cells")
        pieces: list[DomainMap] = []
        for group, cell in enumerate(cells):
            reference_cell = cast("AbstractReferenceCell", cell)
            coordinates = PullBack(self.coordinate_function, cell_map(self.topology, group))
            pieces.append(DomainMap(reference_cell, coordinates, self.ambient_domain))
        return tuple(pieces)

    def _validate(self) -> None:
        from .validation import _require

        f = self.coordinate_function
        topology = self.topology
        _require(
            all(isinstance(K, AbstractReferenceCell) for K in topology.cell_types),
            "mesh cells must be reference cells",
        )
        _require(not f.free_indices, "mesh coordinates must have no free indices")
        _require(
            not f.free_indices
            and (not f.argument_spaces)
            and (f.scalar_type == ScalarType.real)
            and (f.value_shape == (self.ambient_domain.dimension,)),
            "invalid mesh coordinates",
        )
        _require(
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
    @_cached
    def parameterisation(self) -> AbstractExpression:
        from .maps import PullBack

        if not isinstance(self.ambient_domain, Mesh):
            raise TypeError("a topology cell has no parameterisation")
        return PullBack(
            self.ambient_domain.coordinate_function,
            CellMap(self.topology, self.cell_group, self.entry),
        )

    @property
    @_cached
    def free_indices(self) -> tuple[tuple[Index, int | AbstractExtent], ...]:
        if isinstance(self.entry, GatheredIndex):
            return self.entry.free_indices
        if isinstance(self.entry, int):
            return ()
        return ((self.entry, self.topology.num_cells[self.cell_group]),)

    def _validate(self) -> None:
        from .validation import _require

        if type(self.cell_group) is not int:
            raise TypeError("cell group must be an integer")
        _require(0 <= self.cell_group < len(self.topology.cell_types), "invalid cell group")
        self.parameter_domain  # Check that the selected cell is a reference cell.
        size = self.topology.num_cells[self.cell_group]
        if isinstance(self.entry, int):
            _require(
                type(size) is int and 0 <= self.entry < size,
                "cell selection is outside the group",
            )
        elif isinstance(self.entry, GatheredIndex):
            _require(self.entry.extent == size, "cell gather has the wrong extent")
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
        from .validation import _require

        X = self.ambient_domain
        S = self.side_topology
        T = X.topology if isinstance(X, Mesh) else X
        _require(
            all(isinstance(K, AbstractReferenceCell) for K in S.cell_types),
            "side cells must be reference cells",
        )
        _require(
            len(S.sides) == len(S.cell_types)
            and all(
                any((name == self.side and U == T for name, U, _ in sides)) for sides in S.sides
            ),
            "side is absent from a cell group",
        )


_Variance = Literal["covariant", "contravariant"]


@final
class TransformationLaw(Base):
    identifier = "uflx.TransformationLaw"
    # The transformation law of a tensor density: a variance for each of the last axes, the
    # others unchanged, and a weight.
    variances: tuple[_Variance, ...]
    weight: int

    def _validate(self) -> None:
        from .validation import _require

        _require(
            all(v in ("covariant", "contravariant") for v in self.variances), "unknown variance"
        )
        if type(self.weight) is not int:
            raise TypeError("weight must be an integer")


@final
class PushForward(AbstractExpression):
    value_fields = ("expression",)
    identifier = "uflx.PushForward"
    # The field on a mesh of a function on its topology, carried by a transformation law.
    expression: AbstractExpression
    mesh: Mesh
    law: TransformationLaw

    @property
    @_cached
    def value_shape(self) -> tuple[int | AbstractExtent, ...]:
        # A covariant or a contravariant axis has the extent k on the reference cell and n on D.
        shape, p = self.expression.value_shape, len(self.law.variances)
        n = self.mesh.ambient_domain.ambient_coordinate_domain.dimension
        return (*shape[: len(shape) - p], *(n for _ in range(p)))

    @property
    @_cached
    def free_indices(self) -> tuple[tuple[Index, int | AbstractExtent], ...]:
        return self.expression.free_indices

    @property
    @_cached
    def scalar_type(self) -> Literal["real", "complex"]:
        return self.joined_scalar_type(self.expression)

    @property
    @_cached
    def domain(self) -> AbstractDomain | AbstractTopology | None:
        return self.mesh

    @property
    @_cached
    def argument_spaces(
        self,
    ) -> tuple[AbstractFunctionSpace | tuple[AbstractFunctionSpace, ...], ...]:
        return self.expression.argument_spaces

    def _validate(self) -> None:
        from .validation import _require

        _require(
            self.expression.domain in (None, self.mesh.topology),
            "push-forward source differs from the topology",
        )
        p = len(self.law.variances)
        _require(
            len(self.expression.value_shape) >= p
            and (not p or self.expression.value_shape[-p:] == (self.mesh.dimension,) * p),
            "transformed axes have the wrong extent",
        )


@final
class Normal(AbstractExpression):
    value_fields = ()
    is_terminal = True
    identifier = "uflx.Normal"
    side_map: SideMap

    @property
    @_cached
    def value_shape(self) -> tuple[int | AbstractExtent, ...]:

        domain = self.side_map.ambient_domain
        if not isinstance(domain, Mesh):
            raise TypeError("normal requires a mesh side")
        return (domain.ambient_coordinate_domain.dimension,)

    @property
    @_cached
    def scalar_type(self) -> Literal["real", "complex"]:
        return ScalarType.real

    @property
    @_cached
    def domain(self) -> AbstractDomain | AbstractTopology | None:
        return self.side_map.parameter_domain

    @property
    @_cached
    def free_indices(self) -> tuple[tuple[Index, int | AbstractExtent], ...]:
        return ()

    @property
    @_cached
    def argument_spaces(
        self,
    ) -> tuple[AbstractFunctionSpace | tuple[AbstractFunctionSpace, ...], ...]:
        return ()

    def _validate(self) -> None:
        from .validation import _require

        _require(
            isinstance(self.side_map.ambient_domain, Mesh)
            and self.side_map.parameter_domain.dimension
            == self.side_map.ambient_domain.dimension - 1,
            "normal requires a codimension-one mesh side",
        )
