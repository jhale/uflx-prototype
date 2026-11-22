"""Sampling locations and the boundary between interpreted operands and callbacks."""

from collections.abc import Callable
from dataclasses import dataclass

from uflx.nodes import AbstractExpression, AbstractFunctionSpace, AbstractTopology, Mesh


def lookup(bindings, node, label):
    """Library data may be supplied by node or by its runtime label."""
    return bindings[node] if node in bindings else bindings[label]


@dataclass(frozen=True, eq=False)
class Sample:
    """Coordinates and local cell slots supplied for each domain."""

    points: dict
    cells: dict

    def at(self, points, *, cells=None):
        return Sample(self.points | points, self.cells if cells is None else self.cells | cells)

    def slot(self, domain):
        if domain in self.cells:
            return self.cells[domain]
        topology = domain.topology if isinstance(domain, Mesh) else domain
        if isinstance(topology, AbstractTopology) and len(topology.cell_types) != 1:
            raise ValueError("supply a local cell slot in cells for a multi-cell domain")
        return 0


@dataclass(frozen=True)
class BoundExpression:
    """An operand with its lexical bindings captured in read(site, slots)."""

    signature: AbstractExpression | AbstractFunctionSpace
    read: Callable


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
