# Copyright (C) 2026 Jack S. Hale
#
# This file is part of uflx-core-experimental (https://fenics.github.io/uflx/language/)
#
# SPDX-License-Identifier:    MIT

import basix
import numpy as np
import pytest

from uflx_basix.cells import cell, create_topology
from uflx_basix.spaces import function_space


def test_native_variant_precision_and_dof_order_are_part_of_the_space():
    def create(order, variant=basix.LagrangeVariant.gll_warped, dtype=np.float32):
        return basix.create_element(
            basix.ElementFamily.P,
            basix.CellType.interval,
            3,
            variant,
            dof_ordering=order,
            dtype=dtype,
        )

    order = [1, 0, 3, 2]
    element = create(order)
    K = cell(element.cell_type)
    V = function_space(K, element)
    assert V.domain is K
    assert V.element(0) is element
    assert function_space(K, create(order)) is V
    assert function_space(K, create([])) is not V
    assert function_space(K, create(order, dtype=np.float64)) is not V
    assert function_space(K, create(order, basix.LagrangeVariant.equispaced)) is not V


def test_topology_slots_are_distinct_even_for_repeated_cell_types():
    triangle = basix.CellType.triangle
    T = create_topology(triangle, triangle)
    assert T.cell_types == (cell(triangle), cell(triangle))
    assert T[0] is not T[1]
    with pytest.raises(ValueError, match="at least one cell"):
        create_topology()
    with pytest.raises(ValueError, match="same dimension"):
        create_topology(triangle, basix.CellType.interval)


def test_native_elements_follow_topology_slots():
    tri = basix.create_element(basix.ElementFamily.P, basix.CellType.triangle, 1)
    quad = basix.create_tp_element(
        basix.ElementFamily.P, basix.CellType.quadrilateral, 2, basix.LagrangeVariant.equispaced
    )
    T = create_topology(tri.cell_type, quad.cell_type)
    V = function_space(T, (tri, quad))
    assert V.elements == (tri, quad)
    assert V.element(1).dof_ordering == quad.dof_ordering
    assert V.cell_dofs == (tuple(range(tri.dim)), tuple(range(tri.dim, tri.dim + quad.dim)))
    with pytest.raises(ValueError, match="cell type"):
        function_space(T, (quad, tri))
    with pytest.raises(ValueError, match="per cell slot"):
        function_space(T, (tri,))
    with pytest.raises(ValueError, match="common value shape and map type"):
        function_space(
            T,
            (basix.create_element(basix.ElementFamily.RT, tri.cell_type, 1), quad),
        )


def interval_element(*points):
    """A custom discontinuous element with DOFs at the given reference points."""
    base = basix.create_element(
        basix.ElementFamily.P, basix.CellType.interval, len(points) - 1, discontinuous=True
    )
    x = base.x
    x[1][0] = np.array(points).reshape(-1, 1)
    return basix.create_custom_element(
        base.cell_type,
        base.value_shape,
        base.wcoeffs,
        x,
        base.M,
        base.interpolation_nderivs,
        base.map_type,
        base.sobolev_space,
        base.discontinuous,
        base.embedded_subdegree,
        base.embedded_superdegree,
        base.polyset_type,
    )


@pytest.mark.parametrize("custom", [False, True])
def test_element_can_be_rebuilt_from_graph_data(custom):
    if custom:
        element = interval_element(0.375)
    else:
        element = basix.create_element(
            basix.ElementFamily.P, basix.CellType.interval, 1, dof_ordering=[1, 0], dtype=np.float32
        )

    def construction():
        data = function_space(cell(element.cell_type), element).element_data[0]
        return type(data), data.construction_arguments

    cls, args = construction()
    rebuilt = cls(*args).native
    assert rebuilt is not element
    assert rebuilt.dtype == element.dtype and rebuilt.dof_ordering == element.dof_ordering
    np.testing.assert_array_equal(rebuilt.points, element.points)
    np.testing.assert_array_equal(rebuilt.interpolation_matrix, element.interpolation_matrix)
    np.testing.assert_allclose(
        rebuilt.tabulate(1, element.points), element.tabulate(1, element.points)
    )
