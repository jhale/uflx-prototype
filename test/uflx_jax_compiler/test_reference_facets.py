# Copyright (C) 2026 Jack S. Hale
#
# This file is part of uflx-core-experimental (https://fenics.github.io/uflx/language/)
#
# SPDX-License-Identifier:    MIT
"""Standard reference facets with fixed or assembler-selected orientation."""

import basix
import jax
import numpy as np
import pytest

from uflx_basix.cells import cell
from uflx_basix.facets import reference_facet
from uflx_extensions.index_tables import index_table
from uflx_jax_compiler import trace


@pytest.mark.parametrize(
    ("cell_type", "facet", "code", "order"),
    [
        (basix.CellType.interval, 1, 0, (0,)),
        (basix.CellType.triangle, 2, 1, (1, 0)),
        (basix.CellType.quadrilateral, 3, 1, (1, 0)),
        (basix.CellType.tetrahedron, 2, 2, (2, 0, 1)),
        (basix.CellType.tetrahedron, 1, 3, (1, 0, 2)),
        (basix.CellType.hexahedron, 5, 2, (2, 0, 3, 1)),
        (basix.CellType.prism, 1, 4, (3, 2, 1, 0)),
        (basix.CellType.pyramid, 1, 5, (2, 1, 0)),
    ],
)
def test_reference_facet_maps_vertices_in_ufcx_order(cell_type, facet, code, order):
    inclusion, normal = reference_facet(cell(cell_type), facet, code)
    F = inclusion.parameter_domain
    points = basix.cell.geometry(F.cell_type) if F.dimension else np.empty((1, 0))
    mapped = jax.vmap(trace(inclusion.parameterisation, F))(points)
    vertices = basix.cell.topology(cell_type)[-2][facet]
    expected = basix.cell.geometry(cell_type)[vertices]
    np.testing.assert_allclose(mapped, expected[list(order)], atol=1e-7)
    np.testing.assert_allclose(trace(normal)(), basix.cell.facet_outward_normals(cell_type)[facet])


def test_facet_and_permutation_can_be_supplied_at_runtime():
    triangle = basix.CellType.triangle
    facets = index_table("facets", shape=(2,))
    permutations = index_table("permutations", shape=(2,))
    inclusion, normal = reference_facet(cell(triangle), facets[1], permutations[1])
    assert inclusion.free_indices == normal.free_indices == ()
    chart = jax.jit(
        trace(inclusion.parameterisation, inclusion.parameter_domain, facets, permutations)
    )
    outward = jax.jit(trace(normal, facets))
    vertices = basix.cell.geometry(triangle)
    for f, edge in enumerate(basix.cell.topology(triangle)[1]):
        a, b = vertices[edge]
        for p in range(2):
            t = 0.75 if p else 0.25
            result = chart(np.array([0.25]), np.array([0, f]), np.array([0, p], dtype=np.uint8))
            np.testing.assert_allclose(result, (1 - t) * a + t * b)
        np.testing.assert_allclose(
            outward(np.array([0, f])), basix.cell.facet_outward_normals(triangle)[f]
        )

    with pytest.raises(ValueError, match="integer index table"):
        outward(np.array([0.0, 1.0]))
    with pytest.raises(ValueError, match="shape"):
        outward(np.array([1]))


def test_one_facet_array_indexes_cells_with_different_numbers_of_facets():
    facets = index_table("facets", shape=(2,))
    triangle = basix.CellType.triangle
    quadrilateral = basix.CellType.quadrilateral
    _, plus = reference_facet(cell(triangle), facets[0])
    _, minus = reference_facet(cell(quadrilateral), facets[1])
    kernel = jax.jit(trace(plus - minus, facets))
    plus_normals = basix.cell.facet_outward_normals(triangle)
    minus_normals = basix.cell.facet_outward_normals(quadrilateral)
    for i in range(len(plus_normals)):
        for j in range(len(minus_normals)):
            np.testing.assert_allclose(kernel(np.array([i, j])), plus_normals[i] - minus_normals[j])
