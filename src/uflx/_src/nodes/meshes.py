# Copyright (C) 2026 Jack S. Hale
#
# This file is part of uflx-core-experimental (https://fenics.github.io/uflx/language/)
#
# SPDX-License-Identifier:    MIT
"""Meshes, their cells and sides, and mapped fields."""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING, Literal, cast, final

from ..typing import NumericalScalarType, Variance
from .base import Base
from .caching import cached
from .domains import AbstractCoordinateDomain, AbstractDomain, AbstractReferenceCell
from .expressions import AbstractExpression
from .indices import Index
from .scalar_types import ScalarType, numerical_type
from .topology import AbstractTopology

if TYPE_CHECKING:
    from ..typing.nodes import ArgumentSpaces
    from .indices import AbstractExtent
    from .maps import DomainMap


@final
class Mesh(AbstractDomain):
    identifier = "uflx.Mesh"

    coordinate_function: AbstractExpression
    _ambient_domain: AbstractCoordinateDomain

    def __getitem__(self, cell_slot: int) -> CellMap:
        from ..meshes import cell_map

        return cell_map(self, cell_slot)

    def __iter__(self) -> Iterator[CellMap]:
        for slot in range(len(self.topology.cell_types)):
            yield self[slot]

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

    def _validate(self) -> None:
        f = self.coordinate_function
        topology = self.topology
        if any(not isinstance(K, AbstractReferenceCell) for K in topology.cell_types):
            raise ValueError("mesh cells must be reference cells")
        if f.free_indices:
            raise ValueError("mesh coordinates must have no free indices")
        if (
            f.argument_spaces
            or f.scalar_type != ScalarType.real
            or f.value_shape != (self.ambient_domain.dimension,)
        ):
            raise ValueError("invalid mesh coordinates")
        if topology.dimension > self.ambient_domain.dimension:
            raise ValueError("mesh dimension exceeds ambient dimension")


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
        if type(self.cell_slot) is not int:
            raise TypeError("cell slot must be an integer")
        if not 0 <= self.cell_slot < len(self.topology.cell_types):
            raise ValueError("invalid cell slot")
        if not isinstance(self.topology.cell_types[self.cell_slot], AbstractReferenceCell):
            raise ValueError("cell must be a reference cell")


@final
class SideMap(Base):
    identifier = "uflx.SideMap"
    parent: Mesh | AbstractTopology
    source: Mesh | AbstractTopology
    entries: tuple[tuple[int, DomainMap], ...]

    @property
    def source_topology(self) -> AbstractTopology:
        return self.source.topology if isinstance(self.source, Mesh) else self.source

    @property
    def parent_topology(self) -> AbstractTopology:
        return self.parent.topology if isinstance(self.parent, Mesh) else self.parent

    free_indices = ()

    def _validate(self) -> None:
        from .maps import DomainMap

        if not isinstance(cast("object", self.parent), (Mesh, AbstractTopology)):
            raise ValueError("invalid side parent")
        if not isinstance(cast("object", self.source), (Mesh, AbstractTopology)):
            raise ValueError("invalid side source")
        S, T = self.source_topology, self.parent_topology
        if len(self.entries) != len(S.cell_types):
            raise ValueError("one side entry is required per source slot")
        for F, entry in zip(S.cell_types, self.entries, strict=True):
            if len(entry) != 2:
                raise ValueError("a side entry is a parent slot and an inclusion")
            j, A = entry
            if type(j) is not int or not 0 <= j < len(T.cell_types):
                raise ValueError("invalid parent cell slot")
            if not isinstance(F, AbstractReferenceCell):
                raise ValueError("source cell must be a reference cell")
            parent_cell = T.cell_types[j]
            if not isinstance(parent_cell, AbstractReferenceCell):
                raise ValueError("parent cell must be a reference cell")
            if not isinstance(cast("object", A), DomainMap):
                raise ValueError("side inclusion must be a DomainMap")
            if A.parameter_domain != F:
                raise ValueError("side inclusion source differs from source reference cell")
            if A.ambient_domain != parent_cell:
                raise ValueError("side inclusion target differs from parent reference cell")
            if A.free_indices:
                raise ValueError("side inclusions must have no free indices")


@final
class TransformationLaw(Base):
    identifier = "uflx.TransformationLaw"
    # The transformation law of a tensor density: a variance for each of the last axes, the
    # others unchanged, and a weight.
    variances: tuple[Variance, ...]
    weight: int

    def _validate(self) -> None:
        if any(v not in ("covariant", "contravariant") for v in self.variances):
            raise ValueError("unknown variance")
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
    def scalar_type(self) -> NumericalScalarType:
        return numerical_type(self.expression.scalar_type)

    @property
    @cached
    def domain(self) -> AbstractDomain | AbstractTopology | None:
        return self.mesh

    @property
    @cached
    def argument_spaces(self) -> ArgumentSpaces:
        return self.expression.argument_spaces

    def _validate(self) -> None:
        if self.expression.domain not in (None, self.mesh.topology):
            raise ValueError("push-forward source differs from the topology")
        p = len(self.law.variances)
        if len(self.expression.value_shape) < p or (
            p and self.expression.value_shape[-p:] != (self.mesh.dimension,) * p
        ):
            raise ValueError("transformed axes have the wrong extent")


@final
class Conormal(AbstractExpression):
    value_fields = ()
    is_terminal = True
    identifier = "uflx.Conormal"
    side_map: SideMap
    reference_conormals: tuple[AbstractExpression, ...]

    @property
    @cached
    def value_shape(self) -> tuple[int | AbstractExtent, ...]:

        domain = self.side_map.parent
        if not isinstance(domain, Mesh):
            raise TypeError("conormal requires a mesh side")
        return (domain.ambient_coordinate_domain.dimension,)

    @property
    @cached
    def scalar_type(self) -> Literal["real"]:
        return ScalarType.real

    @property
    @cached
    def domain(self) -> AbstractDomain | AbstractTopology | None:
        return self.side_map.source

    @property
    @cached
    def free_indices(self) -> tuple[tuple[Index, int | AbstractExtent], ...]:
        return ()

    @property
    @cached
    def argument_spaces(self) -> ArgumentSpaces:
        return ()

    def _validate(self) -> None:
        if (
            not isinstance(self.side_map.parent, Mesh)
            or self.side_map.source.dimension != self.side_map.parent.dimension - 1
        ):
            raise ValueError("conormal requires a codimension-one mesh side")

        if len(self.reference_conormals) != len(self.side_map.entries):
            raise ValueError("one reference conormal is required per source slot")
        for (_, A), m in zip(self.side_map.entries, self.reference_conormals, strict=True):
            if m.domain not in (None, A.parameter_domain):
                raise ValueError("reference conormal has the wrong domain")
            if m.value_shape != (A.ambient_domain.dimension,):
                raise ValueError("reference conormal has the wrong shape")
            if m.scalar_type != ScalarType.real or m.argument_spaces:
                raise ValueError("reference conormal must be a real field without arguments")
            if m.free_indices:
                raise ValueError("reference conormals must have no free indices")
