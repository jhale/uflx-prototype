# Copyright (C) 2026 Jack S. Hale
#
# This file is part of uflx-core-experimental (https://fenics.github.io/uflx/language/)
#
# SPDX-License-Identifier:    MIT
"""Maps between domains and pull-backs along them."""

from __future__ import annotations

from typing import TYPE_CHECKING, final

from typing_extensions import override

from ..typing import ExpressionScalarType
from .base import Base
from .caching import cached
from .expressions import AbstractExpression
from .scalar_types import ScalarType

if TYPE_CHECKING:
    from ..typing.nodes import ArgumentSpaces
    from .domains import AbstractCoordinateDomain, AbstractDomain
    from .indices import AbstractExtent, Index
    from .meshes import CellMap, SideMap
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

    @override
    def _validate(self) -> None:
        f = self.parameterisation
        if f.domain not in (None, self.parameter_domain):
            raise ValueError("parameterisation domain differs")
        if (
            f.scalar_type != ScalarType.real
            or f.argument_spaces != ()
            or f.value_shape != (self.ambient_domain.dimension,)
        ):
            raise ValueError("invalid coordinate expression")
        if self.parameter_domain.dimension > self.ambient_domain.dimension:
            raise ValueError("map decreases ambient dimension")


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
    def argument_spaces(self) -> ArgumentSpaces:
        return self.expression.argument_spaces

    @override
    def _validate(self) -> None:
        from .validation import merge_indices

        target = self.map.ambient_domain if isinstance(self.map, DomainMap) else self.map.parent
        if self.expression.domain not in (None, target):
            raise ValueError("pull-back target differs from expression domain")
        _ = merge_indices(self.expression.free_indices, self.map.free_indices)
