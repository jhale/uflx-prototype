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
    # Vertex labels are library connectivity, not core side or orientation data.
    _cell_types: tuple
    cells: tuple

    @property
    def cell_types(self):
        return self._cell_types


@final
class TabulatedTopology(AbstractTopology):
    identifier = "toy.TabulatedTopology"
    label: str
    _cell_types: tuple

    @property
    def cell_types(self):
        return self._cell_types
