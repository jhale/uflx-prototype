# Copyright (C) 2026 Jack S. Hale
#
# This file is part of uflx-core-experimental (https://fenics.github.io/uflx/language/)
#
# SPDX-License-Identifier:    MIT
"""Shared transformation laws preserve canonical cell geometry and gradient values."""

from uflx._src.algorithms.substitution import substitute
from uflx._src.nodes import CellMap
from uflx.forms import function
from uflx.geometry import cell_map, create_mesh, jacobian
from uflx.nodes import EuclideanSpace, ScalarType

from ..uflx.spaces import space
from .cells import reference_simplex
from .meshes import grouped_topology


def geometry():
    K = reference_simplex(1)
    T = grouped_topology((K,), (((0, 1),),), ())
    V = space("canonical-coordinates", value_shape=(2,), domain=T, scalar_type=ScalarType.real)
    x, y = function(V, "x"), function(V, "y")
    return T, x, y, create_mesh(x, EuclideanSpace(2))


def test_cell_construction_and_coordinate_replacement_keep_one_representation():
    _T, x, y, D = geometry()
    C = cell_map(D, 0)
    manual = cell_map(D, 0)
    assert isinstance(C, CellMap)
    assert manual is C
    assert manual.parent is D and manual.cell_slot == 0
    assert jacobian(manual) is jacobian(C)
    E = create_mesh(y, EuclideanSpace(2))
    assert substitute(C, {x: y}) is cell_map(E, 0)
    assert substitute(jacobian(C), {x: y}) is jacobian(cell_map(E, 0))
