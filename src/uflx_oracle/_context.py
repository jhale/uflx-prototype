"""Sampling locations and the boundary between interpreted operands and callbacks."""

from collections.abc import Callable
from dataclasses import dataclass

import jax.numpy as jnp

from uflx.nodes import AbstractTopology, Mesh


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
class Operand:
    """An interpreted operand with lexical bindings captured at its definition."""

    read: Callable

    def __call__(self, site, *args):
        return self.read(site, *args)


def bound(value, site):
    if isinstance(value, tuple):
        return tuple(bound(v, site) for v in value)
    if isinstance(value, Operand):
        return lambda *args: bound(value(site, *args), site)
    return value


def call(site, fn, *args):
    if isinstance(fn, Operand):
        return fn(site, *args)
    return fn(*(bound(arg, site) for arg in args))


def sample(field, site, domain):
    result = call(site, field, site.points[domain])
    return jnp.asarray(result) if isinstance(result, (list, tuple)) else result
