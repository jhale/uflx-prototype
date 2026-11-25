# Copyright (C) 2026 Jack S. Hale
#
# This file is part of uflx-core-experimental (https://fenics.github.io/uflx/language/)
#
# SPDX-License-Identifier:    MIT
"""Numerical evaluation of field pull-backs."""

import jax.numpy as jnp

from uflx.nodes import AbstractTopology, CellMap, Mesh, SideMap

from ._geometry import chart_of, side_reference


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
