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
    Integer,
    Tensor,
    Zero,
)


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
    scalar_type = "real"


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
