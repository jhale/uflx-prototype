# Copyright (C) 2026 Jack S. Hale
#
# This file is part of uflx-core-experimental (https://fenics.github.io/uflx/language/)
#
# SPDX-License-Identifier:    MIT
"""Interpolate Basix shape functions at JAX reference coordinates."""

from itertools import combinations, product

import basix
import jax
import jax.numpy as jnp
import numpy as np
from basix import CellType


def _simplex(degree, point):
    """Equispaced simplex nodes and their nodal polynomials in barycentric coordinates."""
    dim = len(point)
    indices = np.array([k for k in product(range(degree + 1), repeat=dim) if sum(k) <= degree])
    alpha = np.column_stack((degree - indices.sum(axis=1), indices))
    bary = jnp.concatenate((jnp.array([1 - point.sum()]), point))
    # For node alpha, the Lagrange polynomial is the product of binomials
    # binom(degree * bary[i], alpha[i]) over the barycentric coordinates.
    # Multiply exactly degree affine factors per node. AD then exposes the
    # constant top derivative to vmap, including the gradient of any P1 field.
    factors = [[(j, k) for j, a in enumerate(row) for k in range(a)] for row in alpha]
    weights = jnp.ones(len(indices), dtype=point.dtype)
    for column in zip(*factors, strict=True):
        axes, k = np.asarray(column).T
        weights = weights * ((degree * bary[axes] - k) / (k + 1))
    return indices / max(degree, 1), weights


def _line(degree, t):
    # Chebyshev nodes stay inside the interval, including on a collapsed pyramid.
    n = degree + 1
    nodes = (1 - np.cos(np.pi * (np.arange(n) + 0.5) / n)) / 2
    others = np.array([np.delete(nodes, i) for i in range(n)])
    weights = jnp.ones(n, dtype=t.dtype)
    for other in others.T:
        weights = weights * ((t - other) / (nodes - other))
    return nodes[:, None], weights


def _macro_simplex(cell, point):
    """Affine charts of the midpoint refinement, with Basix's facet tie-breaking."""
    verts = basix.geometry(cell)
    pieces = [(verts + v) / 2 for v in verts]
    corners = [point.sum() < 0.5, *(point > 0.5)]
    midpoints = [(a + b) / 2 for a, b in combinations(verts, 2)]
    if cell == CellType.triangle:
        pieces.append(midpoints)
    else:
        ab, ac, ad, bc, bd, cd = midpoints
        # The four tetrahedra in the central octahedron share the ac--bd diagonal.
        pieces.extend([[ac, bd, a, b] for a, b in ((ab, ad), (ab, bc), (bc, cd), (cd, ad))])
        x, y, z = point
        corners.extend([((y + z) < 0.5) & ((x + y) < 0.5), (y + z) < 0.5, (x + y) > 0.5])
    slot = jnp.select(corners, tuple(range(len(corners))), default=len(corners))
    pieces = np.asarray(pieces)
    origins, charts = pieces[:, 0], pieces[:, 1:] - pieces[:, :1]
    local = (point - jnp.asarray(origins)[slot]) @ jnp.asarray(np.linalg.inv(charts))[slot]
    return local, origins, charts, slot


def tabulate(element, point):
    """Sample Basix at fixed nodes during tracing. Only interpolation runs in the kernel."""
    return tabulate_rows(element, point, 0)[0]


def tabulate_rows(element, point, order):
    """Interpolate native Basix derivative rows at a runtime reference point."""
    point = jnp.asarray(point, dtype=float)
    dim = len(point)
    if not dim:
        return jnp.asarray(element.tabulate(order, np.empty((1, 0), dtype=element.dtype))[:, 0])

    cell, degree = element.cell_type, element.embedded_superdegree
    simplex = cell in (CellType.triangle, CellType.tetrahedron)
    macro = element.polyset_type == basix.PolysetType.macroedge
    slot = 0
    if macro and simplex:
        point, origins, charts, slot = _macro_simplex(cell, point)
    elif macro:
        halves = np.array(list(product((0, 1), repeat=dim)))
        half = (point > 0.5).astype(int)
        slot = half @ (2 ** np.arange(dim - 1, -1, -1))
        point = 2 * point - half
    elif cell == CellType.pyramid:
        x, y, z = point
        # Basix defines the reciprocal height as one at the apex when tabulating derivatives.
        height = 1 - z + jax.lax.stop_gradient(jnp.where(z == 1, 1, 0))
        point = (x / height, y / height, z)

    if simplex:
        factors = [_simplex(degree, point)]
    elif cell == CellType.prism:
        factors = [_simplex(degree, point[:2]), _line(degree, point[2])]
    else:
        factors = [_line(degree, t) for t in point]

    nodes = np.array([np.concatenate(parts) for parts in product(*(x for x, _ in factors))])
    if macro and simplex:
        nodes = nodes @ charts + origins[:, None, :]
    elif macro:
        nodes = (nodes + halves[:, None, :]) / 2
    elif cell == CellType.pyramid:
        nodes[:, :2] *= 1 - nodes[:, 2:]

    grid = nodes.reshape(-1, dim).astype(element.dtype)
    rows = np.moveaxis(element.tabulate(order, grid), 0, 1)
    shape = (-1, *(len(x) for x, _ in factors), *rows.shape[1:])
    table = jnp.asarray(rows.reshape(shape), dtype=point[0].dtype)[slot]
    for _, weights in factors:
        table = jnp.tensordot(weights, table, axes=1)
    return table
