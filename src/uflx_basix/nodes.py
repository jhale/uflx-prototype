from __future__ import annotations

from functools import cached_property
from math import prod
from typing import final

import basix
import numpy as np
from basix.finite_element import tp_dof_ordering

from uflx.nodes import (
    AbstractExpression,
    AbstractFunctionSpace,
    AbstractReferenceCell,
    AbstractTopology,
    Float,
    ScalarType,
    TransformationLaw,
    Zero,
)
from uflx.tensors import tensor


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
class BasixSpace(AbstractFunctionSpace):
    """A Basix finite element on a reference cell or on each cell of a topology."""

    identifier = "toy.BasixSpace"
    label: str
    _domain: AbstractTopology | BasixCell
    family_value: int
    degree: int
    discontinuous_value: int
    _scalar_type: str
    block_shape: tuple[int, ...]
    argument_spaces = ()

    @property
    def family(self):
        return basix.ElementFamily(self.family_value)

    @property
    def discontinuous(self):
        return bool(self.discontinuous_value)

    def element(self, j):
        return self.elements[j]

    @property
    def topology(self):
        return self.domain

    @cached_property
    def elements(self):
        # The dofs of an element that factors are in the order of its factors, the last fastest.
        out = []
        cells = (self.domain,) if isinstance(self.domain, BasixCell) else self.domain.cell_types
        for K in cells:
            variant = (
                basix.LagrangeVariant.unset
                if self.family in (basix.ElementFamily.HHJ, basix.ElementFamily.Hermite)
                else basix.LagrangeVariant.equispaced
            )
            args = (self.family, K.cell_type, self.degree, variant)
            try:
                order = tp_dof_ordering(*args, discontinuous=self.discontinuous)
            except RuntimeError:
                order = None
            E = basix.create_element(*args, discontinuous=self.discontinuous, dof_ordering=order)
            out.append(E)
        return tuple(out)

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
        return self._domain

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

    @property
    def domain(self):
        return self.space.domain

    @property
    def value_shape(self):
        E = self.space.element(self.cell_slot)
        return (E.dim, *E.value_shape)


def table(values):
    """Convert a numerical Basix array into a UFLx literal."""
    values = np.asarray(values)
    if values.ndim == 0:
        return Float(float(values))
    if 0 in values.shape:
        return Zero(values.shape, (), (), ScalarType.real)
    return tensor(tuple(table(row) for row in values))
