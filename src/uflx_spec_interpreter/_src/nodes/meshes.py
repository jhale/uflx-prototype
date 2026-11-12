"""Meshes, their cells and sides, and mapped fields."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, cast, final

from .base import Base
from .caching import cached
from .domains import AbstractCoordinateDomain, AbstractDomain, AbstractReferenceCell
from .expressions import AbstractExpression
from .scalar_types import ScalarType
from .topology import AbstractTopology

if TYPE_CHECKING:
    from .indices import AbstractExtent
    from .maps import DomainMap
    from .spaces import AbstractFunctionSpace


@final
class Mesh(AbstractDomain):
    identifier = "uflx.Mesh"

    coordinate_function: AbstractExpression
    _ambient_domain: AbstractCoordinateDomain

    @property
    def ambient_domain(self) -> AbstractCoordinateDomain:
        return self._ambient_domain

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
        from .validation import require

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
    parent: AbstractTopology | Mesh
    cell_slot: int
    free_indices = ()

    @property
    def topology(self) -> AbstractTopology:
        return self.parent.topology if isinstance(self.parent, Mesh) else self.parent

    @property
    def reference_cell(self) -> AbstractReferenceCell:
        return cast("AbstractReferenceCell", self.topology.cell_types[self.cell_slot])

    @property
    @cached
    def parameterisation(self) -> AbstractExpression:
        from .maps import PullBack

        if not isinstance(self.parent, Mesh):
            raise TypeError("a topology cell has no parameterisation")
        return PullBack(self.parent.coordinate_function, CellMap(self.topology, self.cell_slot))

    def _validate(self) -> None:
        from .validation import require

        if not isinstance(self.parent, (AbstractTopology, Mesh)):
            raise TypeError("cell parent must be a topology or mesh")
        if type(self.cell_slot) is not int:
            raise TypeError("cell slot must be an integer")
        require(0 <= self.cell_slot < len(self.topology.cell_types), "invalid cell slot")
        require(isinstance(self.reference_cell, AbstractReferenceCell),
                "cell must be a reference cell")


@final
class SideMap(Base):
    identifier = "uflx.SideMap"
    parent: Mesh | AbstractTopology
    restriction_domain: Mesh | AbstractTopology
    side: str
    free_indices = ()

    @property
    def side_topology(self) -> AbstractTopology:
        P = self.restriction_domain
        return P.topology if isinstance(P, Mesh) else P

    def _validate(self) -> None:
        from .validation import require

        require(
            (isinstance(self.parent, Mesh) and isinstance(self.restriction_domain, Mesh))
            or (isinstance(self.parent, AbstractTopology)
                and isinstance(self.restriction_domain, AbstractTopology)),
            "side parent and restriction domain must both be topologies or both be meshes",
        )
        S = self.side_topology
        T = self.parent.topology if isinstance(self.parent, Mesh) else self.parent
        require(len(S.sides) == len(S.cell_types), "side is absent from a cell slot")
        for K, sides in zip(S.cell_types, S.sides, strict=True):
            require(isinstance(K, AbstractReferenceCell), "side cells must be reference cells")
            selected = [j for name, U, j in sides if name == self.side and U == T]
            require(len(selected) == 1, "side must select one parent cell per slot")
            j = selected[0]
            require(type(j) is int and 0 <= j < len(T.cell_types), "invalid parent cell slot")
            require(isinstance(T.cell_types[j], AbstractReferenceCell),
                    "parent cell must be a reference cell")


_Variance = Literal["covariant", "contravariant"]


@final
class TransformationLaw(Base):
    identifier = "uflx.TransformationLaw"
    # The transformation law of a tensor density: a variance for each of the last axes, the
    # others unchanged, and a weight.
    variances: tuple[_Variance, ...]
    weight: int

    def _validate(self) -> None:
        from .validation import require

        require(
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
    @cached
    def value_shape(self) -> tuple[int | AbstractExtent, ...]:
        # A covariant or a contravariant axis has the extent k on the reference cell and n on D.
        shape, p = self.expression.value_shape, len(self.law.variances)
        n = self.mesh.ambient_domain.ambient_coordinate_domain.dimension
        return (*shape[: len(shape) - p], *(n for _ in range(p)))

    @property
    @cached
    def free_indices(self) -> tuple[tuple[Index, int | AbstractExtent], ...]:
        return self.expression.free_indices

    @property
    @cached
    def scalar_type(self) -> Literal["real", "complex"]:
        return self.joined_scalar_type(self.expression)

    @property
    @cached
    def domain(self) -> AbstractDomain | AbstractTopology | None:
        return self.mesh

    @property
    @cached
    def argument_spaces(
        self,
    ) -> tuple[AbstractFunctionSpace | tuple[AbstractFunctionSpace, ...], ...]:
        return self.expression.argument_spaces

    def _validate(self) -> None:
        from .validation import require

        require(
            self.expression.domain in (None, self.mesh.topology),
            "push-forward source differs from the topology",
        )
        p = len(self.law.variances)
        require(
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
    @cached
    def value_shape(self) -> tuple[int | AbstractExtent, ...]:

        domain = self.side_map.parent
        if not isinstance(domain, Mesh):
            raise TypeError("normal requires a mesh side")
        return (domain.ambient_coordinate_domain.dimension,)

    @property
    @cached
    def scalar_type(self) -> Literal["real", "complex"]:
        return ScalarType.real

    @property
    @cached
    def domain(self) -> AbstractDomain | AbstractTopology | None:
        return self.side_map.restriction_domain

    @property
    @cached
    def free_indices(self) -> tuple[tuple[Index, int | AbstractExtent], ...]:
        return ()

    @property
    @cached
    def argument_spaces(
        self,
    ) -> tuple[AbstractFunctionSpace | tuple[AbstractFunctionSpace, ...], ...]:
        return ()

    def _validate(self) -> None:
        from .validation import require

        require(
            isinstance(self.side_map.parent, Mesh)
            and self.side_map.restriction_domain.dimension
            == self.side_map.parent.dimension - 1,
            "normal requires a codimension-one mesh side",
        )
