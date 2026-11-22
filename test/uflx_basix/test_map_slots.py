"""The map contracts use fixed local slots, independently of physical coordinates."""

from operator import getitem

import pytest

from uflx._src.algorithms.substitution import substitute
from uflx._src.nodes import CellMap, Integral, SideMap, Sum
from uflx._src.nodes.topology import AbstractCellType
from uflx._src.rules.restriction import side_cell
from uflx.arithmetic import num
from uflx.calculus import coordinate, integral
from uflx.forms import function
from uflx.geometry import cell_map, domain_map, mesh, pull_back, side_map
from uflx.nodes import EuclideanSpace, ScalarType
from uflx.passes import expand_integrals, expand_pull_backs
from uflx.tensors import index, tensor
from uflx_basix.cells import reference_simplex
from uflx_basix.spaces import space

from .meshes import side_data, static_topology


class PlainCell(AbstractCellType):
    identifier = "test.PlainCell"
    dimension = 2
    num_vertices = 3


def configuration():
    K, F = reference_simplex(2), reference_simplex(1)
    T = static_topology((K, K), ((0, 1, 2), (1, 3, 2)), ())
    S = static_topology((F,), ((1, 2),), ((("+", T, 0), ("-", T, 1)),))
    return K, F, T, S


def test_cell_slots_have_no_indices_and_reconstruct_the_same_selection():
    K, _, T, _ = configuration()
    V = space("coordinates", (2,), T, ScalarType.real)
    x, y = function(V, "x"), function(V, "y")
    D, E = mesh(x, EuclideanSpace(2)), mesh(y, EuclideanSpace(2))
    C = D[1]
    assert C is cell_map(parent=D, cell_slot=1)
    assert T[1] is cell_map(T, 1)
    assert T[1] is not D[1]
    assert C.construction_arguments == (D, 1)
    assert C.parent is D and C.topology is T and C.reference_cell is K
    assert C.free_indices == ()
    assert C.parameterisation is pull_back(x, cell_map(T, 1))
    assert substitute(C, {x: y}) is E[1]
    assert expand_pull_backs(pull_back(coordinate(D), C)) is C.parameterisation
    assert cell_map(T, 0) is not cell_map(T, 1)
    with pytest.raises(TypeError, match="no parameterisation"):
        cell_map(T, 0).parameterisation


@pytest.mark.parametrize("select", [cell_map, getitem])
@pytest.mark.parametrize("slot", [-1, 2])
def test_cell_slot_bounds(select, slot):
    _, _, T, _ = configuration()
    x = function(space("coordinates", (2,), T, ScalarType.real), "x")
    for parent in (T, mesh(x, EuclideanSpace(2))):
        with pytest.raises(ValueError, match="invalid cell slot"):
            select(parent, slot)


def test_cell_selection_rejects_array_indices_and_nonreference_cells():
    _, _, T, _ = configuration()
    x = function(space("coordinates", (2,), T, ScalarType.real), "x")
    for parent in (T, mesh(x, EuclideanSpace(2))):
        for slot in (index("cell"), slice(None), (0,), True):
            with pytest.raises(TypeError):
                parent[slot]
        assert tuple(parent) == (parent[0], parent[1])
    with pytest.raises(TypeError, match="integer"):
        cell_map(T, index("cell"))
    with pytest.raises(TypeError):
        CellMap(T, 0, 0)
    plain = static_topology((PlainCell(),), ((0, 1, 2),), ())
    with pytest.raises(ValueError, match="reference cell"):
        plain[0]


def test_side_map_accepts_mixed_endpoints_and_validates_entries():
    K, _F, T, S = configuration()
    x = function(space("coordinates", (2,), T, ScalarType.real), "x")
    y = function(space("side-coordinates", (2,), S, ScalarType.real), "y")
    D, E = mesh(x, EuclideanSpace(2)), mesh(y, EuclideanSpace(2))
    A, _ = side_data(S, 0, "+")
    for parent, source in ((D, S), (T, E), (D, E), (T, S)):
        relation = side_map(source, (parent[0], A))
        assert relation.parent is parent and relation.source is source
    for slot in (-1, 2):
        with pytest.raises(ValueError, match="parent cell slot"):
            SideMap(T, S, ((slot, A),))
    plain = static_topology((PlainCell(),), ((0, 1, 2),), ())
    with pytest.raises(ValueError, match="parent cell must be a reference cell"):
        SideMap(plain, S, ((0, A),))
    with pytest.raises(ValueError, match="at least one"):
        side_map(S)
    with pytest.raises(ValueError, match="inclusion source"):
        side_map(S, (T[0], domain_map(K, coordinate(K), K)))
    sm = side_map(E, (D[0], A))
    other = mesh(num(2) * y, EuclideanSpace(2))
    assert side_cell(sm, cell_map(E, 0)) == 0
    assert side_cell(sm, cell_map(other, 0)) is None


def test_coordinate_map_families_keep_array_indices_but_mesh_expansion_has_no_cell_sum():
    K, _, T, _ = configuration()
    i = index("parameter")
    psi = tensor((num(1), num(2)))[i] * coordinate(K)
    A = domain_map(K, psi, EuclideanSpace(2))
    assert A.free_indices == ((i, 2),)
    x = function(space("coordinates", (2,), T, ScalarType.real), "x")
    D = mesh(x, EuclideanSpace(2))
    f = tensor((num(2), num(7)))[i]
    out = expand_integrals(integral(f, D))
    assert out.free_indices == ((i, 2),)
    assert isinstance(out, Sum)
    for slot, term in enumerate((out.first, out.second)):
        assert isinstance(term, Integral)
        assert term.expression.first is pull_back(f, D[slot])
