from __future__ import annotations

from typing import final

from .. import oracle
from ..nodes import AbstractExpression, AbstractExtent

# The kernel inputs of the cut cell example, a language extension outside the core.


@final
class RuntimeExtent(AbstractExtent):
    identifier = "runtime.RuntimeExtent"
    label: str
    opaque = ("label",)


@final
class RuntimeArray(AbstractExpression):
    identifier = "runtime.RuntimeArray"
    label: str
    value_shape: tuple
    opaque = ("label", "value_shape")
    free_indices = ()
    arity = 0
    support = None


def runtime_extent(label):
    return RuntimeExtent(label)


def runtime_array(label, value_shape):
    return RuntimeArray(label, tuple(value_shape))


oracle.RULES[RuntimeArray] = lambda e, env: lambda vals: env[e]
