# Copyright (C) 2026 Jack S. Hale
#
# This file is part of uflx-core-experimental (https://fenics.github.io/uflx/language/)
#
# SPDX-License-Identifier:    MIT
"""Standard reference-facet inclusions and outward normals from Basix."""

from operator import index

import basix
import numpy as np

from uflx.calculus import coordinate
from uflx.geometry import domain_map
from uflx.nodes import GatheredIndex

from .cells import cell
from .nodes import BasixCell, table


def _vertex_orders(cell_type, facet_type, num_vertices):
    """Vertex order for each UFCx quadrature permutation on this facet type."""
    count = {1: 1, 2: 2, 3: 6, 4: 8}[num_vertices]
    orders = np.tile(np.arange(num_vertices, dtype=np.int32), (count, 1))
    if count > 1:
        p1 = basix.create_element(basix.ElementFamily.P, cell_type, 1)
        for code in range(1, count):
            # Inverse DOF permutation maps oriented quadrature points into the cell.
            p1.permute_subentity_closure_inv(orders[code], code, facet_type)
    return orders


def reference_facet(
    K: BasixCell, local_facet: int | GatheredIndex, permutation: int | GatheredIndex = 0
):
    """Return the oriented facet inclusion into K and its outward reference unit normal.

    The local facet number and UFCx quadrature permutation can be integers or gathered
    indices. A runtime facet number requires all facets to have the same reference type.
    """
    if K.dimension == 0:
        raise ValueError("a point has no facets")
    facets = basix.cell.topology(K.cell_type)[-2]
    types = basix.cell.subentity_types(K.cell_type)[-2]
    normals = basix.cell.facet_outward_normals(K.cell_type)
    if isinstance(local_facet, GatheredIndex):
        if len(set(types)) != 1:
            raise ValueError("a runtime facet number requires a single reference facet type")
        facet_type = types[0]
    else:
        local_facet = index(local_facet)
        if not 0 <= local_facet < len(facets):
            raise IndexError("local facet number is outside the reference cell")
        facet_type = types[local_facet]
        facets = [facets[local_facet]]
        normals = normals[[local_facet]]
        local_facet = 0

    F = cell(facet_type)
    orders = _vertex_orders(K.cell_type, facet_type, len(facets[0]))
    if not isinstance(permutation, GatheredIndex):
        permutation = index(permutation)
        if not 0 <= permutation < len(orders):
            raise IndexError("quadrature permutation is outside the facet's orientation range")
        orders = orders[[permutation]]
        permutation = 0

    vertices = basix.cell.geometry(K.cell_type)
    points = vertices[np.asarray(facets)[:, orders]]
    # Basix reference facets are affine. The first vertex is the origin and the
    # next dim(F) vertices give its coordinate directions, including on quadrilaterals.
    origins = points[:, :, 0]
    chart = table(origins)[local_facet, permutation]
    for d in range(F.dimension):
        tangents = points[:, :, d + 1] - origins
        chart = chart + table(tangents)[local_facet, permutation] * coordinate(F)[d]
    return domain_map(F, chart, K), table(normals)[local_facet]
