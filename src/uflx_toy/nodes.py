from __future__ import annotations

from functools import cached_property
from typing import final

from uflx_spec_interpreter._src.nodes import (
    AbstractDomain,
    AbstractExpression,
    AbstractExtent,
    AbstractFunctionSpace,
    AbstractIndexTable,
    AbstractReferenceCell,
    AbstractTopology,
    Integer,
    ScalarType,
    Tensor,
    Zero,
)

from .inclusions import side_tables, vertex_rows


@final
class IndexTable(AbstractIndexTable):
    identifier = "toy.IndexTable"
    # The toy knows the values, which the specification leaves to a library.
    values: tuple
    _extent: int

    @cached_property
    def shape(self):
        shape, x = [], self.values
        while isinstance(x, tuple):
            shape.append(len(x))
            x = x[0]
        return tuple(shape)

    @property
    def extent(self):
        return self._extent


@final
class RuntimeExtent(AbstractExtent):
    identifier = "toy.RuntimeExtent"
    # The extent of the cut cell example, known only at assembly.
    label: str


@final
class RuntimeArray(AbstractExpression):
    is_terminal = True
    identifier = "toy.RuntimeArray"
    # A kernel input of the cut cell example, such as quadrature data.
    label: str
    _value_shape: tuple
    free_indices = ()
    argument_spaces = ()
    domain = None
    scalar_type = ScalarType.real

    @property
    def value_shape(self):
        return self._value_shape


@final
class Space(AbstractFunctionSpace):
    identifier = "toy.Space"
    # A space of fields, whose functions the oracle reads from its environment.
    name: str
    _value_shape: tuple
    domain_data: AbstractDomain | tuple
    _scalar_type: str
    argument_spaces = ()

    @property
    def domain(self):
        return self.domain_data or None

    @property
    def value_shape(self):
        return self._value_shape

    @property
    def scalar_type(self):
        return self._scalar_type


@final
class VertexCoordinates(AbstractExpression):
    is_terminal = True
    identifier = "toy.VertexCoordinates"
    # The coordinate function of a mesh from the coordinates of its vertices, of degree 1 on each
    # cell, which the toy evaluates with the shape functions of the vertices.
    topology: AbstractTopology
    vertices: tuple
    free_indices = ()
    argument_spaces = ()
    scalar_type = ScalarType.real

    @property
    def value_shape(self):
        return (len(self.vertices[0]),)

    @property
    def domain(self):
        return self.topology


@final
class ReferenceSimplex(AbstractReferenceCell):
    identifier = "toy.ReferenceSimplex"
    # The unit simplex, with vertex 0 at the origin and vertex v the unit vector e_(v-1).
    _dimension: int

    @property
    def num_vertices(self):
        return self.dimension + 1

    @property
    def vertices(self):
        n = self.dimension
        if n == 0:
            return Zero((1, 0), (), (), ScalarType.real)
        return Tensor(
            tuple(
                Tensor(tuple(Integer(1 if k == v - 1 else 0) for k in range(n)))
                for v in range(n + 1)
            )
        )

    @property
    def dimension(self):
        return self._dimension


@final
class StaticTopology(AbstractTopology):
    identifier = "toy.StaticTopology"
    # Cells as vertex numbers, with the side tables computed from the vertex numbers.
    _cell_types: tuple
    cells: tuple
    _sides: tuple

    @property
    def num_cells(self):
        return tuple(len(c) for c in self.cells)

    @property
    def sides(self):
        return self._sides

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

    @property
    def cell_types(self):
        return self._cell_types


@final
class TabulatedTopology(AbstractTopology):
    identifier = "toy.TabulatedTopology"
    # The cells and vertex positions of each side as a mesh library computes them.
    label: str
    _cell_types: tuple
    _num_cells: tuple
    _sides: tuple
    tables: tuple

    @property
    def sides(self):
        return self._sides

    def side_tables(self, i, side):
        names = [s[0] for s in self.sides[i]]
        cells, positions = self.tables[i][names.index(side)]
        return side_tables(self, i, side, cells, positions)

    @property
    def cell_types(self):
        return self._cell_types

    @property
    def num_cells(self):
        return self._num_cells
