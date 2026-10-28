from functools import cache
from typing import NamedTuple

from uflx_spec_interpreter._src.nodes import AbstractExtent
from uflx_spec_interpreter._src.substitution import expressions
from uflx_toy.nodes import RuntimeArray


class Basis(NamedTuple):
    # The label of a basis table at points known only at assembly, which runtime_tables fills.
    space: object
    piece: int
    points: object
    derivative: int


def nodes(e):
    # Each node of the graph of e once.
    seen = {}

    def walk(n):
        if n not in seen:
            seen[n] = None
            for c in expressions(n):
                walk(c)

    walk(e)
    return list(seen)


@cache
def reads_input(e):
    return isinstance(e, RuntimeArray) or any(map(reads_input, expressions(e)))


def kernel_inputs(e):
    # The arrays that a consumer passes by label, with the label of an extent known only at
    # assembly in place of its length, which the consumer passes under that label too. The
    # basis tables are left out, because runtime_tables computes them from their points.
    arrays = [n for n in nodes(e) if isinstance(n, RuntimeArray)]
    tables = [n.label for n in arrays if isinstance(n.label, Basis)]
    arrays += [n for b in tables for n in nodes(b.points) if isinstance(n, RuntimeArray)]
    return {
        n.label: tuple(m.label if isinstance(m, AbstractExtent) else m for m in n.value_shape)
        for n in arrays
        if not isinstance(n.label, Basis)
    }
