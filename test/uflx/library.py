"""Concrete library data for testing UFLx extension contracts."""

from functools import cached_property
from typing import final

from uflx.nodes import (
    AbstractExpression,
    AbstractExtent,
    AbstractIndexTable,
    AbstractTopology,
    ScalarType,
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


def index_table(values, extent):
    return IndexTable(values, extent)


def runtime_extent(label):
    return RuntimeExtent(label)


def runtime_array(label, value_shape):
    return RuntimeArray(label, tuple(value_shape))


def static_topology(cell_types, cells):
    return StaticTopology(tuple(cell_types), tuple(cells))


def tabulated_topology(label, cell_types):
    return TabulatedTopology(label, tuple(cell_types))
