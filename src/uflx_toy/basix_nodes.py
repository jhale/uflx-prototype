from __future__ import annotations

from functools import cached_property
from typing import final

import basix
import numpy as np
from basix.finite_element import tp_dof_ordering

from uflx_spec_interpreter import tensor
from uflx_spec_interpreter._src.nodes import (
    AbstractFunctionSpace,
    AbstractReferenceCell,
    DecomposedDomain,
    Float,
    ScalarType,
    Zero,
)


@final
class BasixCell(AbstractReferenceCell):
    identifier = "toy.BasixCell"
    # A reference cell of Basix, whose vertices and facet normals Basix tabulates.
    cell_type: basix.CellType
    opaque = ("cell_type",)

    @property
    def dimension(self):
        return len(basix.cell.topology(self.cell_type)) - 1

    @property
    def num_vertices(self):
        return len(basix.cell.topology(self.cell_type)[0])

    @property
    def vertices(self):
        if self.dimension == 0:
            return Zero((1, 0), (), ())
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
    # A Basix element on each cell type of a mesh, with the global dofs of each cell.
    label: str
    domain: DecomposedDomain
    family: basix.ElementFamily
    degree: int
    discontinuous: bool
    scalar_type: ScalarType
    opaque = ("label", "domain", "family", "degree", "discontinuous", "scalar_type")
    argument_spaces = ()

    def element(self, j):
        return self.elements[j][0]

    def factors(self, j):
        # The interval elements whose product is the element of piece j, or None.
        return self.elements[j][1]

    @cached_property
    def elements(self):
        # The dofs of an element that factors are in the order of its factors, the last fastest.
        out = []
        for piece in self.domain.pieces:
            args = (self.family, piece.parameter_domain.cell_type, self.degree, EQUISPACED)
            try:
                order = tp_dof_ordering(*args, discontinuous=self.discontinuous)
            except RuntimeError:
                order = None
            E = basix.create_element(*args, discontinuous=self.discontinuous, dof_ordering=order)
            out.append((E, E.get_tensor_product_representation()[0] if order else None))
        return tuple(out)

    @property
    def value_shape(self):
        return tuple(self.element(0).value_shape)

    @cached_property
    def dofmap(self):
        # Cells share the dofs of their shared vertices, edges and faces, so the space is
        # continuous. A discontinuous element has all of its dofs inside the cell.
        T, first, count = self.domain.topology, {}, 0
        dofmap = []
        for j, cells in enumerate(T.cells):
            E = self.element(j)
            entities = basix.cell.topology(T.cell_types[j].cell_type)
            tdim = len(entities) - 1
            rows = []
            for c, vertices in enumerate(cells):
                row = [0] * E.dim
                for d, on_dim in enumerate(E.entity_dofs):
                    for n, local in enumerate(on_dim):
                        if d == tdim:
                            key = (j, c)
                        else:
                            # One dof on an edge or a face needs no orientation.
                            assert d == 0 or len(local) <= 1, "dof orientation is not covered"
                            key = (d, frozenset(vertices[v] for v in entities[d][n]))
                        if key not in first:
                            first[key], count = count, count + len(local)
                        for m, dof in enumerate(local):
                            row[dof] = first[key] + m
                rows.append(tuple(row))
            dofmap.append(tuple(rows))
        return tuple(dofmap)

    @property
    def num_dofs(self):
        return 1 + max(d for rows in self.dofmap for row in rows for d in row)


EQUISPACED = basix.LagrangeVariant.equispaced


def table(values):
    return tensor(tuple(tensor(tuple(Float(float(x)) for x in row)) for row in np.asarray(values)))
