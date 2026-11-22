from uflx._src import nodes
from uflx._src.meshes import cell_map
from uflx._src.nodes import CellMap, Coordinate, GatheredIndex, Indexed, PullBack
from uflx._src.passes import pullbacks
from uflx_basix.cells import reference_simplex
from uflx_basix.nodes import IndexTable

from .integrals import D, c


def test_the_static_type_is_computed_once():
    e = Indexed(Coordinate(D), (0,))
    assert e.value_shape is e.value_shape
    assert e.free_indices is e.free_indices


def test_a_pull_back_leaves_a_gathered_entry_unchanged():
    g = GatheredIndex(IndexTable((1, 0), 2), (c,))
    out = pullbacks.expand_pull_backs(PullBack(Indexed(Coordinate(D), (g,)), cell_map(D, 0)))
    assert isinstance(out, Indexed) and out.indices == (g,) and out.indices[0] is g
    # The coordinates of a cell of a mesh are the read of its coordinate function on the cell.
    assert out.expression == cell_map(D, 0).parameterisation


def test_a_reference_cell_is_a_coordinate_domain_and_a_cell_type():
    K = reference_simplex(2)
    assert isinstance(K, nodes.AbstractCoordinateDomain) and isinstance(K, nodes.AbstractCellType)
    assert K.dimension == 2 and K.num_vertices == 3 and K.vertices.value_shape == (3, 2)
    assert reference_simplex(0).vertices.value_shape == (1, 0)
    assert K != nodes.EuclideanSpace(2) and reference_simplex(0) != nodes.EuclideanSpace(0)


def test_cell_maps_select_local_slots_of_the_mesh():
    cell = cell_map(D, 0)
    assert cell == CellMap(D, 0)
    assert cell.free_indices == ()
    assert cell.reference_cell == D.topology.cell_types[0]
    x = D.coordinate_function
    assert cell.parameterisation == PullBack(x, CellMap(x.domain, 0))
    assert cell_map(D, 1).parameterisation == PullBack(x, CellMap(x.domain, 1))
