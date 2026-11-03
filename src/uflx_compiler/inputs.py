from __future__ import annotations

from typing import final

from uflx_spec_interpreter._src.nodes import AbstractExtent, Base
from uflx_spec_interpreter._src.substitution import _expressions, _memoized
from uflx_toy.nodes import RuntimeArray


@final
class Basis(Base):
    identifier = "compiler.Basis"
    # The label of a basis table at points known only at assembly, which runtime_tables fills.
    space: object
    piece: int
    points: object
    direction: int | None
    component: int


def arrays(arrays, e):
    # The runtime arrays that e reads, in the order of a walk from e.
    own = {e: None} if isinstance(e, RuntimeArray) else {}
    return own | {n: None for c in _expressions(e) for n in arrays(c)}


def runtime_arrays(e):
    return list(_memoized(arrays)(e))


def kernel_inputs(e):
    # The arrays that a consumer passes by label, with the label of an extent known only at
    # assembly in place of its length, which the consumer passes under that label too. The
    # basis tables are left out, because runtime_tables computes them from their points.
    found = runtime_arrays(e)
    tables = [n.label for n in found if isinstance(n.label, Basis)]
    found = [n for n in found if n.label not in tables]
    found += [n for b in tables for n in runtime_arrays(b.points)]
    return {
        n.label: tuple(m.label if isinstance(m, AbstractExtent) else m for m in n.value_shape)
        for n in found
    }
