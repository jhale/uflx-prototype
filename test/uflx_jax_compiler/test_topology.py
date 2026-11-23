import numpy as np

from uflx_jax_compiler import trace

from ..uflx_basix.integrals import I as interval
from ..uflx_basix.integrals import K
from ..uflx_basix.meshes import grouped_topology, side_data, side_map


def test_the_side_data_of_the_diagonal_of_the_example():
    # The example of Restriction, with the square numbered ((0, 1, 2), (1, 3, 2)).
    square = grouped_topology((K,), (((0, 1, 2), (1, 3, 2)),), ())
    diagonal = grouped_topology((interval,), (((1, 2),),), ((("+", square, 0), ("-", square, 0)),))
    expected = {"+": ((0,), lambda t: (1 - t, t)), "-": ((1,), lambda t: (0, t))}
    for side, (cells, inclusion) in expected.items():
        A, m = side_data(diagonal, 0, side)
        assert side_map(square, diagonal, side).entries[0][0] == cells[0]
        assert A.parameter_domain == interval
        for t in (0.0, 0.4, 1.0):
            point = trace(A.parameterisation, interval)(np.array([t]))
            assert np.allclose(np.real(point), inclusion(t))
        normal = np.real(trace(m, interval)(np.array([0.5])))
        # Out of K, from the centroid (1/3, 1/3) through the facet.
        assert normal @ (inclusion(0.5) - np.array([1, 1]) / 3) > 0
