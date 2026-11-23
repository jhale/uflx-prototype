from __future__ import annotations

from functools import cached_property
from math import prod
from typing import final

import basix
import numpy as np

from uflx.nodes import (
    AbstractExpression,
    AbstractFunctionSpace,
    AbstractReferenceCell,
    AbstractTopology,
    CellMap,
    Float,
    Mesh,
    ScalarType,
    TransformationLaw,
    Zero,
)
from uflx.tensors import tensor

from ._elements import BuiltinElement, CustomElement


@final
class BasixCell(AbstractReferenceCell):
    """A UFLx reference cell with Basix geometry and vertex numbering."""

    identifier = "toy.BasixCell"
    cell_type_value: int

    @property
    def cell_type(self):
        return basix.CellType(self.cell_type_value)

    @property
    def dimension(self):
        return len(basix.cell.topology(self.cell_type)) - 1

    @property
    def num_vertices(self):
        return len(basix.cell.topology(self.cell_type)[0])

    @property
    def vertices(self):
        if self.dimension == 0:
            return Zero((1, 0), (), (), ScalarType.real)
        return table(basix.cell.geometry(self.cell_type))


@final
class BasixTopology(AbstractTopology):
    """An ordered collection of local cells with Basix reference geometry."""

    identifier = "uflx_basix.BasixTopology"
    _cell_types: tuple[BasixCell, ...]

    @property
    def cell_types(self) -> tuple[BasixCell, ...]:
        return self._cell_types

    def _validate(self):
        if not self.cell_types:
            raise ValueError("a topology requires at least one cell")
        if not all(isinstance(K, BasixCell) for K in self.cell_types):
            raise TypeError("expected Basix reference cells")
        if any(K.dimension != self.dimension for K in self.cell_types[1:]):
            raise ValueError("topology cells must have the same dimension")


@final
class BasixSpace(AbstractFunctionSpace):
    """A Basix finite element on a reference cell, a mesh cell, or a topology."""

    identifier = "toy.BasixSpace"
    support: AbstractTopology | BasixCell | CellMap
    element_data: tuple[BuiltinElement | CustomElement, ...]
    _scalar_type: str
    block_shape: tuple[int, ...]
    argument_spaces = ()

    def _validate(self):
        if isinstance(self.support, CellMap) and not isinstance(self.support.parent, Mesh):
            raise ValueError("a cell-bound space requires a physical mesh cell")
        if not isinstance(self.domain, (BasixCell, AbstractTopology)):
            raise ValueError("expected a Basix reference cell or a topology")
        cells = (self.domain,) if isinstance(self.domain, BasixCell) else self.domain.cell_types
        if not cells or len(self.element_data) != len(cells):
            raise ValueError("supply one Basix element per cell slot")
        if any(n <= 0 for n in self.block_shape):
            raise ValueError("block shape extents must be positive")
        first = self.element(0)
        for K, E in zip(cells, self.elements, strict=True):
            if not isinstance(K, BasixCell) or K.cell_type != E.cell_type:
                raise ValueError("element cell type must match its domain cell")
            if self.block_shape and E.value_shape:
                raise ValueError("blocked spaces require a scalar base element")
            if E.value_shape != first.value_shape or E.map_type != first.map_type:
                raise ValueError("elements must have a common value shape and map type")

    def on_cell(self, cell: CellMap) -> BasixSpace:
        """Select this space's element on one physical cell."""
        if not isinstance(cell, CellMap) or not isinstance(cell.parent, Mesh):
            raise ValueError("expected a physical mesh cell")
        if self.cell is not None and self.cell != cell:
            raise ValueError("space is already restricted to a different cell")
        if self.domain == cell.topology:
            data = (self.element_data[cell.cell_slot],)
        elif self.domain == cell.reference_cell:
            data = self.element_data
        else:
            raise ValueError("cell does not belong to the space's domain")
        return BasixSpace(cell, data, self.scalar_type, self.block_shape)

    @property
    def cell(self) -> CellMap | None:
        return self.support if isinstance(self.support, CellMap) else None

    def element(self, j):
        return self.elements[j]

    @property
    def topology(self):
        return self.cell.topology if self.cell is not None else self.domain

    @cached_property
    def elements(self):
        return tuple(data.native for data in self.element_data)

    @property
    def block_size(self):
        return prod(self.block_shape)

    @property
    def value_shape(self):
        return self.block_shape or tuple(self.element(0).value_shape)

    @property
    def law(self):
        # The transformation law of the map type of the element, as Basix names the six maps.
        match self.element(0).map_type:
            case basix.MapType.identity:
                return TransformationLaw((), 0)
            case basix.MapType.covariantPiola:
                return TransformationLaw(("covariant",), 0)
            case basix.MapType.contravariantPiola:
                return TransformationLaw(("contravariant",), 1)
            case basix.MapType.doubleCovariantPiola:
                return TransformationLaw(("covariant", "covariant"), 0)
            case basix.MapType.doubleContravariantPiola:
                return TransformationLaw(("contravariant", "contravariant"), 2)
            case basix.MapType.L2Piola:
                return TransformationLaw((), 1)
        raise ValueError(f"no transformation law for {self.element(0).map_type}")

    @cached_property
    def cell_dofs(self):
        # Packed cell-local coefficients. Vertex labels never identify field DOFs.
        rows, offset = [], 0
        for E in self.elements:
            size = E.dim * self.block_size
            rows.append(tuple(range(offset, offset + size)))
            offset += size
        return tuple(rows)

    @property
    def num_dofs(self):
        return sum(len(row) for row in self.cell_dofs)

    @property
    def domain(self):
        return self.support.reference_cell if isinstance(self.support, CellMap) else self.support

    @property
    def scalar_type(self):
        return self._scalar_type


@final
class BasixBasis(AbstractExpression):
    """The unblocked basis of one cell, extended by zero on other cell slots."""

    identifier = "toy.BasixBasis"
    is_terminal = True
    space: BasixSpace
    cell_slot: int
    free_indices = ()
    argument_spaces = ()
    scalar_type = ScalarType.real

    def _validate(self):
        if self.space.cell is not None and self.cell_slot != self.space.cell.cell_slot:
            raise ValueError("basis slot differs from the restricted space")

    @property
    def domain(self):
        return self.space.topology

    @property
    def element(self):
        return self.space.element(0 if self.space.cell is not None else self.cell_slot)

    @property
    def value_shape(self):
        return (self.element.dim, *self.element.value_shape)


def table(values):
    """Convert a numerical Basix array into a UFLx literal."""
    values = np.asarray(values)
    if values.ndim == 0:
        return Float(float(values))
    if 0 in values.shape:
        return Zero(values.shape, (), (), ScalarType.real)
    return tensor(tuple(table(row) for row in values))
