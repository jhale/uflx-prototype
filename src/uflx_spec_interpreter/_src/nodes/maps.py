"""Maps between domains and pull-backs along them."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, final

from .base import AbstractExpression, Base
from .interning import _cached
from .scalar_types import ScalarType

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
    @_cached
    def free_indices(self) -> tuple[tuple[Index, int | AbstractExtent], ...]:
        return self.parameterisation.free_indices

    def _validate(self) -> None:
        from .validation import _require

        f = self.parameterisation
        _require(f.domain in (None, self.parameter_domain), "parameterisation domain differs")
        _require(
            f.scalar_type == ScalarType.real
            and (not f.argument_spaces)
            and (f.value_shape == (self.ambient_domain.dimension,)),
            "invalid coordinate expression",
        )
        _require(
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
    @_cached
    def value_shape(self) -> tuple[int | AbstractExtent, ...]:
        return self.expression.value_shape

    @property
    @_cached
    def free_indices(self) -> tuple[tuple[Index, int | AbstractExtent], ...]:
        return tuple((dict(self.expression.free_indices) | dict(self.map.free_indices)).items())

    @property
    @_cached
    def scalar_type(self) -> Literal["real", "complex"]:
        return self.joined_scalar_type(self.expression)

    @property
    @_cached
    def domain(self) -> AbstractDomain | AbstractTopology | None:
        return self.map.parameter_domain

    @property
    @_cached
    def argument_spaces(
        self,
    ) -> tuple[AbstractFunctionSpace | tuple[AbstractFunctionSpace, ...], ...]:
        return self.expression.argument_spaces

    def _validate(self) -> None:
        from .validation import _merge_indices, _require

        _require(
            self.expression.domain in (None, self.map.ambient_domain),
            "pull-back target differs from expression domain",
        )
        _merge_indices(self.expression.free_indices, self.map.free_indices)
