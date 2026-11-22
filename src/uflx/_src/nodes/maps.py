"""Maps between domains and pull-backs along them."""

from __future__ import annotations

from typing import TYPE_CHECKING, final

from .base import Base
from .caching import cached
from .expressions import AbstractExpression
from .scalar_types import ExpressionScalarType, ScalarType

if TYPE_CHECKING:
    from .domains import AbstractCoordinateDomain, AbstractDomain
    from .indices import AbstractExtent, Index
    from .meshes import CellMap, SideMap
    from .spaces import AbstractFunctionSpace
    from .topology import AbstractTopology


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
        from .validation import require

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
class PullBack(AbstractExpression):
    value_fields = ("expression",)
    identifier = "uflx.PullBack"
    expression: AbstractExpression
    map: DomainMap | SideMap | CellMap

    @property
    @cached
    def value_shape(self) -> tuple[int | AbstractExtent, ...]:
        return self.expression.value_shape

    @property
    @cached
    def free_indices(self) -> tuple[tuple[Index, int | AbstractExtent], ...]:
        return tuple((dict(self.expression.free_indices) | dict(self.map.free_indices)).items())

    @property
    @cached
    def scalar_type(self) -> ExpressionScalarType:
        return self.expression.scalar_type

    @property
    @cached
    def domain(self) -> AbstractDomain | AbstractTopology | None:
        from .meshes import CellMap, SideMap

        if isinstance(self.map, CellMap):
            return self.map.reference_cell
        if isinstance(self.map, SideMap):
            return self.map.source
        return self.map.parameter_domain

    @property
    @cached
    def argument_spaces(
        self,
    ) -> tuple[AbstractFunctionSpace | tuple[AbstractFunctionSpace, ...], ...]:
        return self.expression.argument_spaces

    def _validate(self) -> None:
        from .validation import merge_indices, require

        target = self.map.ambient_domain if isinstance(self.map, DomainMap) else self.map.parent
        require(
            self.expression.domain in (None, target),
            "pull-back target differs from expression domain",
        )
        merge_indices(self.expression.free_indices, self.map.free_indices)
