# Copyright (C) 2026 Jack S. Hale
#
# This file is part of uflx-core-experimental (https://fenics.github.io/uflx/language/)
#
# SPDX-License-Identifier:    MIT

import basix
import numpy as np

from uflx._src.meshes import is_cell
from uflx._src.nodes import CellMap
from uflx.forms import function
from uflx.geometry import cell_map, create_mesh, pull_back
from uflx.nodes import EuclideanSpace
from uflx.tensors import tensor
from uflx_basix.cells import cell
from uflx_basix.spaces import local_space

from .meshes import grouped_topology, side_domain, side_map

P = basix.ElementFamily.P
K = cell(basix.CellType.triangle)
# The unit square as two triangles, with the vertices of square(K) of test_basix.
T = grouped_topology((K,), (((0, 1, 2), (0, 2, 3)),), ())
VERTICES = np.array([(0, 0), (1, 0), (1, 1), (0, 1)], dtype=float)


def coordinates(degree, discontinuous=False):
    element = basix.create_element(P, K.cell_type, degree, discontinuous=discontinuous)
    X = local_space(T, element)
    x0, x1 = function(X, "x0"), function(X, "x1")
    return X, x0, x1, tensor((x0, x1))


def p1_mesh():
    # The library gathers shared vertex coordinates into the two local cells.
    X, x0, x1, x = coordinates(1)
    assert X.cell_dofs == ((0, 1, 2), (3, 4, 5))
    packed = VERTICES[np.asarray(T.cells).ravel()]
    return create_mesh(x, EuclideanSpace(2)), {x0: packed[:, 0], x1: packed[:, 1]}


def test_a_mesh_from_a_coordinate_function_has_the_topology_and_the_cells_of_the_draft():
    D, _ = p1_mesh()
    x = D.coordinate_function
    assert D.topology == T and x.domain == T
    C = cell_map(D, 1)
    assert C.parameterisation == pull_back(x, CellMap(T, 1)) and is_cell(C)
    assert cell_map(T, 0) == CellMap(T, 0)


def test_a_side_domain_is_the_mesh_of_the_trace_of_the_coordinate_function():
    D, _ = p1_mesh()
    I = cell(basix.CellType.interval)
    S = grouped_topology((I,), (((0, 2),),), ((("+", T, 0), ("-", T, 0)),))
    E = side_domain(D, S)
    assert E.topology == S
    assert E.coordinate_function == pull_back(
        D.coordinate_function,
        side_map(T, S),
    )
