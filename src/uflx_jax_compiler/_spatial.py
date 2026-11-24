# Copyright (C) 2026 Jack S. Hale
#
# This file is part of uflx-core-experimental (https://fenics.github.io/uflx/language/)
#
# SPDX-License-Identifier:    MIT
"""Numerical meanings of spatial derivatives, field mappings and conormals."""

import jax
import jax.numpy as jnp

from uflx.nodes import AbstractTopology, CellMap, Mesh, SideMap

from ._geometry import (
    chart_of,
    reference_point,
    side_reference,
    transformed,
)
from ._tensors import inverse


def curl(g):
    if g.shape[-1] == 2:
        return g[..., 1, 0] - g[..., 0, 1]
    return jnp.stack(
        (g[..., 2, 1] - g[..., 1, 2], g[..., 0, 2] - g[..., 2, 0], g[..., 1, 0] - g[..., 0, 1]),
        axis=-1,
    )


def pullback_site(mapping, site, value):
    if isinstance(mapping, SideMap):
        _, slot, _, _, point = side_reference(mapping, site, value)
        parent = mapping.parent
        if isinstance(parent, Mesh):
            point = chart_of(parent[slot], site, value)(point)
        return site.at({parent: point}, cells={parent: slot})
    if isinstance(mapping, CellMap):
        parent = mapping.parent
        if isinstance(parent, AbstractTopology):
            cell = mapping.reference_cell
            tables = {parent: site.tabulations[cell]} if cell in site.tabulations else None
            return site.at(
                {parent: site.points[cell]}, cells={parent: mapping.cell_slot}, tabulations=tables
            )
        point = jnp.real(value(mapping.parameterisation))
        return site.at({parent: point}, cells={parent: mapping.cell_slot})
    point = jnp.real(value(mapping.parameterisation))
    return site.at({mapping.ambient_domain: point})


def push_forward(field, mesh, law, site, value):
    slot = site.slot(mesh)
    chart = chart_of(mesh[slot], site, value)
    ref = reference_point(chart, site.points[mesh], mesh.dimension)
    topology = mesh.topology
    site = site.at({topology: ref}, cells={topology: slot})
    return transformed(field(site), law, jax.jacfwd(chart)(ref))


def conormal(side, conormals, site, value):
    slot, parent_slot, inclusion, ref, target = side_reference(side, site, value)
    J = jax.jacfwd(chart_of(side.parent[parent_slot], site, value))(target)
    site = site.at({inclusion.parameter_domain: ref})
    normal = jnp.real(value(conormals[slot], at=site))
    vector = inverse(J).T @ normal
    return vector / jnp.linalg.norm(vector)
