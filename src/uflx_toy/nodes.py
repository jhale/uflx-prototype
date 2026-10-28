from __future__ import annotations

from functools import cached_property
from typing import final

from uflx_spec_interpreter._src.nodes import (
    AbstractDomain,
    AbstractExpression,
    AbstractExtent,
    AbstractFunctionSpace,
    AbstractIndexValue,
    AbstractReferenceCell,
    AbstractTopology,
    Integer,
    ScalarType,
    Tensor,
    Zero,
)

from .inclusions import side_tables, vertex_rows


@final
class IndexTable(AbstractIndexValue):
    identifier = "toy.IndexTable"
    # The toy knows the values, which the specification leaves to a library.
    values: tuple
    extent: int
    opaque = ("values", "extent")

    @cached_property
    def shape(self):
        shape, x = [], self.values
        while isinstance(x, tuple):
            shape.append(len(x))
            x = x[0]
        return tuple(shape)


@final
class RuntimeExtent(AbstractExtent):
    identifier = "toy.RuntimeExtent"
    # The extent of the cut cell example, known only at assembly.
    label: str
    opaque = ("label",)


@final
class RuntimeArray(AbstractExpression):
    identifier = "toy.RuntimeArray"
    # A kernel input of the cut cell example, such as quadrature data.
    label: str
    value_shape: tuple
    opaque = ("label", "value_shape")
    free_indices = ()
    argument_spaces = ()
    support = None
    scalar_type = ScalarType.real


@final
class Space(AbstractFunctionSpace):
    identifier = "toy.Space"
    # A space of fields, whose functions the oracle reads from its environment.
    name: str
    value_shape: tuple
    domain: AbstractDomain | None
    scalar_type: str
    opaque = ("name", "value_shape", "domain", "scalar_type")
    argument_spaces = ()


@final
class ReferenceSimplex(AbstractReferenceCell):
    identifier = "toy.ReferenceSimplex"
    # The unit simplex, with vertex 0 at the origin and vertex v the unit vector e_(v-1).
    dimension: int
    opaque = ("dimension",)

    @property
    def num_vertices(self):
        return self.dimension + 1

    @property
    def vertices(self):
        n = self.dimension
        if n == 0:
            return Zero((1, 0), (), ())
        return Tensor(
            tuple(
                Tensor(tuple(Integer(1 if k == v - 1 else 0) for k in range(n)))
                for v in range(n + 1)
            )
        )


@final
class StaticTopology(AbstractTopology):
    identifier = "toy.StaticTopology"
    # Cells as vertex numbers, with the side tables computed from the vertex numbers.
    cell_types: tuple
    cells: tuple
    sides: tuple
    opaque = ("cell_types", "cells", "sides")

    @property
    def num_cells(self):
        return tuple(len(c) for c in self.cells)

    def side_tables(self, i, side):
        # The cells of the side that contain the vertices, and the vertices of K_j there.
        # Of the cells around a cell, the r-th side of the same group gets the r-th lowest.
        named = next(s for s in self.sides[i] if s[0] == side)
        (_, T, j) = named
        rank = [s for s in self.sides[i] if s[1:] == (T, j)].index(named)
        K = T.cell_types[j]
        chosen, local = [], []
        for vertices in self.cells[i]:
            around = [c for c, cell in enumerate(T.cells[j]) if set(vertices) <= set(cell)]
            c = around[min(rank, len(around) - 1)]
            chosen.append(c)
            local.append([T.cells[j][c].index(v) for v in vertices])
        cells = IndexTable(tuple(chosen), T.num_cells[j])
        positions = Tensor(tuple(vertex_rows(K, vs) for vs in local))
        # A cell that tabulates its facet normals reads them by the local facet of each cell.
        facets = None
        if hasattr(K, "facet_normals") and self.cell_types[i].dimension == K.dimension - 1:
            facets = IndexTable(tuple(K.local_facet(vs) for vs in local), K.num_facets)
        return side_tables(self, i, side, cells, positions, facets)


@final
class TabulatedTopology(AbstractTopology):
    identifier = "toy.TabulatedTopology"
    # The cells and vertex positions of each side as a mesh library computes them.
    label: str
    cell_types: tuple
    num_cells: tuple
    sides: tuple
    tables: tuple
    opaque = ("label", "cell_types", "num_cells", "sides", "tables")

    def side_tables(self, i, side):
        names = [s[0] for s in self.sides[i]]
        cells, positions = self.tables[i][names.index(side)]
        return side_tables(self, i, side, cells, positions)
