"""Physical-cell interpolation includes both component transformations."""

import basix
import numpy as np
import pytest
from test_interpolation import sample
from test_nested_quadrature import edge_sampling_rule
from testutil import vertex_mesh

from uflx_compiler import compile_form, evaluate
from uflx_spec_interpreter import (
    cell_map,
    directional_derivative,
    function,
    integral,
    num,
    pull_back,
    push_forward,
)
from uflx_toy.basix_cells import basix_cell
from uflx_toy.basix_spaces import basix_space
from uflx_toy.interpolation import interpolate
from uflx_toy.topology import static_topology


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize(
    "family",
    [
        basix.ElementFamily.P,
        basix.ElementFamily.N1E,
        basix.ElementFamily.RT,
        basix.ElementFamily.HHJ,
    ],
)
def test_physical_interpolation_matches_basix(family, reverse):
    K = basix_cell(basix.CellType.triangle)
    T = static_topology((K,), ((0, 1, 2),))
    origin = np.array([0.7, -0.4])
    J = np.array([[1.7, 0.4], [-0.2, 1.3]])
    if reverse:
        J = J[:, ::-1]
    detJ, Jinv = np.linalg.det(J), np.linalg.inv(J)
    D = vertex_mesh(T, [origin, origin + J[:, 0], origin + J[:, 1]])
    C = cell_map(D, 0)
    V = basix_space("target", T, family, 1, shape=(2,) if family == basix.ElementFamily.P else ())
    U = basix_space("source", T, basix.ElementFamily.P, 2, shape=V.value_shape)
    u, h = function(U, "u"), function(U, "h")
    E, source = V.element(0), U.element(0)
    rule = edge_sampling_rule(E) if family == basix.ElementFamily.N1E else None
    f = push_forward(u, D, U.law)
    interpolant = interpolate(f, V, cell=C, sampling_rule=rule)
    assert interpolate(f, V, sampling_rule=rule) == interpolant
    assert interpolant.domain == D and interpolant.value_shape == V.value_shape
    rng = np.random.default_rng(17)
    coefficients = rng.normal(size=(source.dim, U.block_size))
    direction = rng.normal(size=coefficients.shape)
    env = {"coefficients:u": coefficients.ravel(), "coefficients:h": direction.ravel()}
    sampling_points, matrix = (E.points, E.interpolation_matrix) if rule is None else rule

    def expected(coefficients, points):
        physical = source.tabulate(0, sampling_points)[0, :, :, 0] @ coefficients
        if V.block_shape:
            dofs = matrix @ physical
            return E.tabulate(0, points)[0, :, :, 0] @ dofs
        reference = E.pull_back(physical[None], J[None], np.array([detJ]), Jinv[None])[0]
        dofs = matrix @ reference.T.ravel()
        values = np.einsum("qac,a->qc", E.tabulate(0, points)[0], dofs)
        return E.push_forward(values[None], J[None], np.array([detJ]), Jinv[None])[0]

    points = np.array([[0.1, 0.2], [0.6, 0.1], [0.2, 0.8]])
    actual = np.array(
        [
            evaluate(compile_form(sample(pull_back(interpolant, C), p, K), 4), env).reshape(-1)
            for p in points
        ]
    )
    np.testing.assert_allclose(actual, expected(coefficients, points), atol=1e-11)

    if family == basix.ElementFamily.N1E:
        energy = integral((interpolant[0] ** num(2) + interpolant[1] ** num(2)) / num(2), D)
        derivative = directional_derivative(energy, u, u, h)
        second = directional_derivative(derivative, u, u, h)
        q, weights = basix.make_quadrature(basix.CellType.triangle, 4)
        values, variations = expected(coefficients, q), expected(direction, q)
        wanted = abs(detJ) * np.array(
            [
                np.einsum("q,qc,qc->", weights, values, values) / 2,
                np.einsum("q,qc,qc->", weights, values, variations),
                np.einsum("q,qc,qc->", weights, variations, variations),
            ]
        )
        for expression, value in zip((energy, derivative, second), wanted, strict=True):
            np.testing.assert_allclose(
                evaluate(compile_form(expression, 4), env), value, atol=1e-11
            )
