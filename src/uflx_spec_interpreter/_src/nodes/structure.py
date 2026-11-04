"""Inspect node dependencies and expression operands."""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

from .base import Base
from .expressions import AbstractExpression
from .types import Construction


def operands(e: Base, names: Sequence[str] | None = None) -> list[Base]:
    # The nodes on the edges of e, in order, through tuples, or those of the named arguments.
    return list(_edges(e) if names is None else nodes_in(e, names))


def _edges(e: Base) -> tuple[Base, ...]:
    # All construction dependencies, including spaces, maps, labels and literal parts.
    memo = vars(e)
    if "edges" not in memo:
        memo["edges"] = nodes_in(e, e.argument_names)
    return cast("tuple[Base, ...]", memo["edges"])


def nodes_in(e: Base, names: Sequence[str]) -> tuple[Base, ...]:
    out: list[Base] = []

    def walk(x: Construction) -> None:
        if isinstance(x, Base):
            out.append(x)
        elif isinstance(x, tuple):
            for y in x:
                walk(y)

    for name in names:
        walk(getattr(e, name))
    return tuple(out)


def expression_operands(e: Base) -> list[AbstractExpression]:
    # The operands of e that are expressions, which a grammar restricts.
    return [x for x in nodes_in(e, e.value_fields) if isinstance(x, AbstractExpression)]
