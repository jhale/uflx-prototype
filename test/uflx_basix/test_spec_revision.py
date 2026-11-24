"""Regression examples in the order of the revised language chapters."""

from __future__ import annotations

import pytest

from uflx.algorithms import replace
from uflx.forms import function
from uflx.geometry import cell_map, create_mesh
from uflx.nodes import EuclideanSpace, ScalarType, Zero
from uflx.tensors import gathered_index, index, indexed

from ..uflx.library import index_table, runtime_array
from ..uflx.spaces import space
from .cells import reference_simplex
from .meshes import grouped_topology


def test_gathered_indices_follow_direct_indices_in_static_order():
    i, j = index("i"), index("j")
    g = gathered_index(index_table((0, 1), 2), (i,))
    A = runtime_array("A", (2, 3))
    assert indexed(A, (g, j)).free_indices == ((j, 3), (i, 2))
    with pytest.raises(ValueError):
        indexed(A, (j, j))
    with pytest.raises(IndexError):
        indexed(A, (2, 0))


def test_mesh_cells_retain_selection_after_geometry_substitution():
    T = grouped_topology((reference_simplex(1),), (((0, 1), (2, 3)),), ())
    V = space("coordinates", (1,), T, ScalarType.real)
    x, y = function(V, "x"), function(V, "y")
    D, E = create_mesh(x, EuclideanSpace(1)), create_mesh(y, EuclideanSpace(1))
    C = cell_map(D, 1)
    assert C.cell_slot == 1 and C.free_indices == ()
    assert C.parameterisation.map is cell_map(T, 1)
    assert replace(C, {x: y}) is cell_map(E, 1)
    with pytest.raises(ValueError):
        cell_map(D, -1)
    with pytest.raises(ValueError):
        cell_map(D, 2)


def test_substituting_into_a_zero_selects_its_family_indices():
    i, j = index("zero-i"), index("zero-j")
    z = Zero((), ((i, 2),), (), ScalarType.real)
    assert replace(z, {i: 0}).free_indices == ()
    assert replace(z, {i: j}).free_indices == ((j, 2),)
    g = gathered_index(index_table((1, 0, 1), 2), (j,))
    assert replace(z, {i: g}).free_indices == ((j, 3),)
