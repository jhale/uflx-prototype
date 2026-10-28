import numpy as np
from test_integrals import DIAGONAL, SQUARE, D, K, c
from test_integrals import I as interval

from uflx_oracle import evaluate
from uflx_spec_interpreter import cell_index, function, num, restriction, side_domain, side_map
from uflx_spec_interpreter._src.nodes import SideMap
from uflx_toy.spaces import space
from uflx_toy.topology import static_topology


def test_the_side_tables_of_the_diagonal_of_the_example():
    # The example of Restriction, with the square numbered ((0, 1, 2), (1, 3, 2)).
    square = static_topology((K,), (((0, 1, 2), (1, 3, 2)),), ())
    diagonal = static_topology((interval,), (((1, 2),),), ((("+", square, 0), ("-", square, 0)),))
    expected = {"+": ((0,), [[(1, 0), (0, 1)]]), "-": ((1,), [[(0, 0), (0, 1)]])}
    for side, (cells, positions) in expected.items():
        table, coordinates = diagonal.side_tables(0, side)
        assert table.values == cells and table.extent == 2
        assert np.allclose(np.real(evaluate(coordinates, {})(())), positions)


def test_a_mesh_and_its_sides_have_topologies():
    assert c == cell_index(SQUARE, 0) and D.topology == SQUARE
    E = side_domain(D, DIAGONAL)
    assert E.topology == DIAGONAL and E.dimension == 1 and E.ambient_domain == D.ambient_domain


def test_a_restriction_is_a_pull_back_by_a_side_map():
    E = side_domain(D, DIAGONAL)
    u = function(space("V", domain=D), "u")
    assert restriction(num(1), E, "+") == num(1)
    f = restriction(u, E, "+")
    assert f.domain == side_map(D, E, "+") and isinstance(f.domain, SideMap) and f.support == E
