# Copyright (C) 2026 Jack S. Hale
#
# This file is part of uflx-core-experimental (https://fenics.github.io/uflx/language/)
#
# SPDX-License-Identifier:    MIT
"""Inspect node dependencies and expression operands."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, cast

from .base import Base
from .caching import cached
from .expressions import AbstractExpression

if TYPE_CHECKING:
    from ..typing.nodes import Construction
    from .domains import AbstractDomain
    from .topology import AbstractTopology


def operands(e: Base, names: Sequence[str] | None = None) -> list[Base]:
    """Return immediate node dependencies in construction order, optionally selecting fields."""
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
    """Return expression operands in the node's declared value fields."""
    return [x for x in nodes_in(e, e.value_fields) if isinstance(x, AbstractExpression)]


@cached
def common_operand_domain(e: AbstractExpression) -> AbstractDomain | AbstractTopology | None:
    """Return the shared domain of the value operands that have one."""
    domains = {operand.domain for operand in expression_operands(e)}
    domains.discard(None)
    if len(domains) > 1:
        raise ValueError("operand domains differ")
    return next(iter(domains), None)
