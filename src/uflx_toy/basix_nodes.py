from __future__ import annotations

from functools import cached_property
from math import prod
from typing import final

import basix
import numpy as np
from basix.finite_element import tp_dof_ordering

from uflx_spec_interpreter import tensor
from uflx_spec_interpreter._src.nodes import (
    AbstractFunctionSpace,
    AbstractReferenceCell,
    AbstractTopology,
    Float,
    ScalarType,
    TransformationLaw,
    Zero,
)


@final
class BasixCell(AbstractReferenceCell):
    identifier = "toy.BasixCell"
    # A reference cell of Basix, whose vertices and facet normals Basix tabulates.
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

    @property
    def num_facets(self):
        return len(basix.cell.topology(self.cell_type)[self.dimension - 1])

    def local_facet(self, local_vertices):
        # The number of the facet of the cell with these local vertices, in the order of Basix.
        facets = basix.cell.topology(self.cell_type)[self.dimension - 1]
        return next(f for f, v in enumerate(facets) if set(v) == set(local_vertices))

    @property
    def facet_normals(self):
        # The outward normal of each facet, a constant table with no support.
        return table(basix.cell.facet_outward_normals(self.cell_type))


@final
class BasixSpace(AbstractFunctionSpace):
    identifier = "toy.BasixSpace"
    # A Basix element on each cell slot, with independent cell-local coefficients. Every
    # element of Basix that the toy builds is Piola equivalent, so the space is on a topology.
    label: str
    _domain: AbstractTopology
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
        return self.elements[j][0]

    def factors(self, j):
        # The interval elements whose product is the element of piece j, or None.
        return self.elements[j][1]

    @property
    def topology(self):
        return self.domain

    @cached_property
    def elements(self):
        # The dofs of an element that factors are in the order of its factors, the last fastest.
        out = []
        for K in self.domain.cell_types:
            variant = (
                basix.LagrangeVariant.unset
                if self.family == basix.ElementFamily.HHJ
                else EQUISPACED
            )
            args = (self.family, K.cell_type, self.degree, variant)
            try:
                order = tp_dof_ordering(*args, discontinuous=self.discontinuous)
            except RuntimeError:
                order = None
            E = basix.create_element(*args, discontinuous=self.discontinuous, dof_ordering=order)
            out.append((E, E.get_tensor_product_representation()[0] if order else None))
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
        for E, _ in self.elements:
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


EQUISPACED = basix.LagrangeVariant.equispaced


def table(values):
    return tensor(tuple(tensor(tuple(Float(float(x)) for x in row)) for row in np.asarray(values)))
