# Copyright (C) 2026 Jack S. Hale
#
# This file is part of uflx-core-experimental (https://fenics.github.io/uflx/language/)
#
# SPDX-License-Identifier:    MIT
"""Sampling locations and the boundary between interpreted operands and callbacks."""

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from types import ModuleType

from uflx.nodes import AbstractExpression, AbstractFunctionSpace, AbstractTopology, Mesh


def lookup(bindings, node, label=None):
    """Library data may be supplied by node or by its runtime label."""
    if node in bindings:
        return bindings[node]
    return bindings[getattr(node, "label", node) if label is None else label]


@dataclass(frozen=True, eq=False)
class PointTabulation:
    center: object
    rows: dict
    derivative_order: int = 0


@dataclass(frozen=True, eq=False)
class Sample:
    """Coordinates and local cell slots supplied for each domain."""

    points: dict
    cells: dict
    tabulations: dict = field(default_factory=dict)
    reference: ModuleType | None = None
    shared_jacobians: frozenset[AbstractExpression] = frozenset()
    runtime_table_elements: frozenset = frozenset()

    def require_reference(self, operation):
        if self.reference is None:
            raise ValueError(f"{operation} requires lowering before tracing")
        return self.reference

    def at(self, points, *, cells=None, tabulations=None):
        # Moving to a different point invalidates the old quadrature rows for that domain.
        tables = self.tabulations
        if tables:
            tables = {d: t for d, t in tables.items() if d not in points}
        if tabulations:
            tables = tables | tabulations
        return Sample(
            self.points | points,
            self.cells if cells is None else self.cells | cells,
            tables,
            self.reference,
            self.shared_jacobians,
            self.runtime_table_elements,
        )

    def differentiating(self, domain, point):
        table = self.tabulations.get(domain)
        tables = (
            {domain: replace(table, derivative_order=table.derivative_order + 1)} if table else None
        )
        return self.at({domain: point}, tabulations=tables)

    def slot(self, domain):
        if domain in self.cells:
            return self.cells[domain]
        topology = domain.topology if isinstance(domain, Mesh) else domain
        if isinstance(topology, AbstractTopology) and len(topology.cell_types) != 1:
            raise ValueError("select a local cell slot with a pull-back to its reference cell")
        return 0


@dataclass(frozen=True)
class BoundExpression:
    """An operand with its lexical bindings captured in read(site, slots)."""

    signature: AbstractExpression | AbstractFunctionSpace
    read: Callable
    needs_eager_samples: bool = True
    is_basis_field: bool = False


def as_argument(value, site):
    """Expose an interpreted operand to a numerical function at this site."""
    if isinstance(value, tuple):
        return tuple(as_argument(item, site) for item in value)
    if not isinstance(value, BoundExpression):
        return value

    def apply(*slots):
        domain = value.signature.domain
        if domain is None:
            return value.read(site, slots)
        return lambda point: value.read(site.at({domain: point}), slots)

    return apply if value.signature.argument_spaces else apply()
