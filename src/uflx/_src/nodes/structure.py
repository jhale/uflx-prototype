# Copyright (C) 2026 Jack S. Hale
#
# This file is part of uflx-core-experimental (https://fenics.github.io/uflx/language/)
#
# SPDX-License-Identifier:    MIT
"""Inspect node dependencies and expression operands."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING, cast

from .base import Base, construction_field
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


def construction_nodes(value: Construction) -> Iterator[Base]:
    """Iterate nodes inside construction tuples in field order."""
    pending = [value]
    while len(pending) > 0:
        item = pending.pop()
        if isinstance(item, Base):
            yield item
        elif isinstance(item, tuple):
            pending.extend(reversed(item))


def nodes_in(e: Base, names: Sequence[str]) -> tuple[Base, ...]:
    return tuple(node for name in names for node in construction_nodes(construction_field(e, name)))


def value_operands(e: Base) -> tuple[AbstractExpression, ...]:
    """Ordered expression edges, shared by the passes over this immutable node."""
    memo = vars(e)
    if "uflx.value_operands" not in memo:
        memo["uflx.value_operands"] = tuple(
            x for x in nodes_in(e, e.value_fields) if isinstance(x, AbstractExpression)
        )
    return cast("tuple[AbstractExpression, ...]", memo["uflx.value_operands"])


def expression_operands(e: Base) -> list[AbstractExpression]:
    """Return expression operands in the node's declared value fields."""
    return list(value_operands(e))


@cached
def common_operand_domain(e: AbstractExpression) -> AbstractDomain | AbstractTopology | None:
    """Return the shared domain of the value operands that have one."""
    domains = {operand.domain for operand in expression_operands(e)}
    domains.discard(None)
    if len(domains) > 1:
        raise ValueError("operand domains differ")
    return next(iter(domains), None)
