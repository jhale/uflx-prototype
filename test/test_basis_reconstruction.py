"""Library basis data with core reconstruction and spatial mapping."""

import basix
import numpy as np
import pytest
from test_interpolation import sample

from uflx._src.nodes import PushForward
from uflx.forms import function
from uflx.geometry import cell_map, mesh, pull_back, push_forward
from uflx.nodes import EuclideanSpace
from uflx_basix.basix_cells import basix_cell
from uflx_basix.basix_spaces import basix_space
from uflx_basix.interpolation import interpolate
from uflx_basix.topology import static_topology
from uflx_compiler import compile_form, evaluate
from uflx_oracle import evaluate as oracle


@pytest.mark.parametrize("curvature", [0.0, 0.3])
def test_interpolation_on_two_cells_with_variable_jacobians(curvature):
    K = basix_cell(basix.CellType.triangle)
    T = static_topology((K, K), ((0, 1, 2), (3, 4, 5)))
    G = basix_space("geometry", T, basix.ElementFamily.P, 2, shape=(2,))
    U = basix_space("source", T, basix.ElementFamily.P, 2, shape=(2,))
    V = basix_space("target", T, basix.ElementFamily.N1E, 1)
    x, u = function(G, "x"), function(U, "u")
    D = mesh(x, EuclideanSpace(2))
    f = push_forward(u, D, U.law)
    result = interpolate(f, V)
    local = interpolate(f, V, cell=cell_map(D, 0))
    assert isinstance(local, PushForward)
    assert result.domain == D
    rng = np.random.default_rng(81)
    coefficients = rng.normal(size=(2, U.element(0).dim, 2))
    matrices = [np.array([[1.7, 0.4], [-0.2, 1.3]]), np.array([[0.8, -0.2], [0.3, 1.2]])]
    origins = [np.array([0.2, -0.3]), np.array([2.0, 0.1])]

    def jacobians(points, j):
        J = np.broadcast_to(matrices[j], (len(points), 2, 2)).copy()
        J[:, 0, 0] += curvature * points[:, 1]
        J[:, 0, 1] += curvature * points[:, 0]
        return J

    gp = G.element(0).points
    geometry = np.array(
        [
            origins[j]
            + gp @ matrices[j].T
            + np.column_stack((curvature * gp[:, 0] * gp[:, 1], np.zeros(len(gp))))
            for j in range(2)
        ]
    )
    env = {"coefficients:x": geometry.ravel(), "coefficients:u": coefficients.ravel()}
    points = np.array([[0.1, 0.2], [0.6, 0.1]])
    E = V.element(0)
    for j in range(2):
        J = jacobians(E.points, j)
        values = U.element(j).tabulate(0, E.points)[0, :, :, 0] @ coefficients[j]
        reference = E.pull_back(values[:, None, :], J, np.linalg.det(J), np.linalg.inv(J))[:, 0]
        dofs = E.interpolation_matrix @ reference.T.ravel()
        reference_output = np.einsum("qac,a->qc", E.tabulate(0, points)[0], dofs)
        J = jacobians(points, j)
        wanted = E.push_forward(
            reference_output[:, None, :], J, np.linalg.det(J), np.linalg.inv(J)
        )[:, 0]
        for expression, expected in (
            (result, wanted),
            (local, wanted if j == 0 else np.zeros_like(wanted)),
        ):
            actual = [
                evaluate(compile_form(sample(pull_back(expression, cell_map(D, j)), p, K), 4), env)
                for p in points
            ]
            np.testing.assert_allclose(actual, expected, atol=1e-11)
            point_value = sample(pull_back(expression, cell_map(D, j)), points[0], K)
            np.testing.assert_allclose(oracle(point_value, env)(()), expected[0], atol=1e-6)
