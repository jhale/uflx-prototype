import numpy as np
from test_integrals import DIAGONAL, SQUARE, D, K, c
from test_integrals import I as interval

from uflx_oracle import at, evaluate
from uflx_spec_interpreter import cell_index, function, pull_back, side_domain, side_map
from uflx_spec_interpreter._src.nodes import SideMap
from uflx_toy.spaces import space
from uflx_toy.topology import static_topology


def test_the_side_tables_of_the_diagonal_of_the_example():
    # The example of Restriction, with the square numbered ((0, 1, 2), (1, 3, 2)).
    square = static_topology((K,), (((0, 1, 2), (1, 3, 2)),), ())
    diagonal = static_topology((interval,), (((1, 2),),), ((("+", square, 0), ("-", square, 0)),))
    e = cell_index(diagonal, 0)
    expected = {"+": ((0,), lambda t: (1 - t, t)), "-": ((1,), lambda t: (0, t))}
    for side, (cells, inclusion) in expected.items():
        table, A, m = diagonal.side_tables(0, side)
        assert table.values == cells and table.extent == 2 and A.parameter_domain == interval
        for t in (0.0, 0.4, 1.0):
            point = at(
                interval, np.array([t]), lambda A=A: evaluate(A.parameterisation, {}, {e: 0})(())
            )
            assert np.allclose(np.real(point), inclusion(t))
        normal = np.real(at(interval, np.array([0.5]), lambda m=m: evaluate(m, {}, {e: 0})(())))
        # Out of K, from the centroid (1/3, 1/3) through the facet.
        assert normal @ (inclusion(0.5) - np.array([1, 1]) / 3) > 0


def test_a_mesh_and_its_sides_have_topologies():
    assert c == cell_index(SQUARE, 0) and D.topology == SQUARE
    E = side_domain(D, DIAGONAL)
    assert E.topology == DIAGONAL and E.dimension == 1 and E.ambient_domain == D.ambient_domain


def test_a_restriction_is_a_pull_back_by_a_side_map():
    E = side_domain(D, DIAGONAL)
    u = function(space("V", domain=D), "u")
    f = pull_back(u, side_map(D, E, "+"))
    assert isinstance(f.domain, SideMap) and f.support == E and f.free_indices == ()


def test_a_side_domain_takes_the_side_on_the_topology_of_the_domain():
    # As for a mesh and a submesh, the first side is on another topology.
    other = static_topology((K,), (((0, 1, 2),),), ())
    S = static_topology(
        (interval,), (((0, 1), (1, 2)),), ((("submesh", other, 0), ("mesh", SQUARE, 0)),)
    )
    E = side_domain(D, S)
    assert E.topology == S and E.ambient_domain == D.ambient_domain
