from __future__ import annotations

from functools import cached_property
from typing import final

from uflx_spec_interpreter.nodes import AbstractExpression, AbstractExtent, AbstractIndexValue


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
    arity = 0
    support = None
