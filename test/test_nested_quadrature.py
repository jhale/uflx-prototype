"""Interpolation inside cell integrals must retain valid domains during lowering."""

import basix
import numpy as np

from uflx_compiler import compile_form, evaluate
from uflx_spec_interpreter import (
    EuclideanSpace,
    cell_map,
    directional_derivative,
    domain_map,
    function,
    integral,
    num,
    pull_back,
    tensor,
)
from uflx_toy.basix_cells import basix_cell
from uflx_toy.basix_spaces import basix_space
from uflx_toy.interpolation import interpolate
from uflx_toy.topology import static_topology


def dot(a, b):
    """Contract two real vectors."""
    return a[0] * b[0] + a[1] * b[1]


def point_value(f, point, cell):
    """Construct a domain-free symbolic point evaluation."""
    position = tensor(tuple(num(float(x)) for x in point))
    return integral(pull_back(f, domain_map(EuclideanSpace(0), position, cell)), EuclideanSpace(0))


def edge_sampling_rule(target):
    """Use two-point Gauss quadrature for exact quadratic edge moments."""
    vertices = basix.geometry(basix.CellType.triangle)
    parameters, weights = np.polynomial.legendre.leggauss(2)
    points = []
    matrix = np.zeros((3, 12))
    for edge, (a, b) in enumerate(((1, 2), (0, 2), (0, 1))):
        tangent = vertices[b] - vertices[a]
        points.extend(vertices[a] + (parameters[:, None] + 1) / 2 * tangent)
        for c in range(2):
            matrix[edge, c * 6 + edge * 2 : c * 6 + edge * 2 + 2] = weights / 2 * tangent[c]
    points = np.array(points)
    basis = target.tabulate(0, points)[0].transpose(2, 0, 1).reshape(12, 3)
    return points, np.linalg.solve(matrix @ basis, matrix)


def reference_values(source, coefficients, points):
    """Independently integrate moments and reconstruct using Basix's basis."""
    target = basix.create_element(basix.ElementFamily.N1E, basix.CellType.triangle, 1)
    vertices = basix.geometry(basix.CellType.triangle)
    parameters, weights = np.polynomial.legendre.leggauss(5)
    moments, moment_matrix = [], []
    for a, b in ((1, 2), (0, 2), (0, 1)):
        tangent = vertices[b] - vertices[a]
        samples = vertices[a] + (parameters[:, None] + 1) / 2 * tangent
        values = source.tabulate(0, samples)[0, :, :, 0] @ coefficients
        moments.append(np.einsum("q,qc,c->", weights / 2, values, tangent))
        target_values = target.tabulate(0, samples)[0]
        moment_matrix.append(np.einsum("q,qac,c->a", weights / 2, target_values, tangent))
    dofs = np.linalg.solve(moment_matrix, moments)
    return np.einsum("qac,a->qc", target.tabulate(0, points)[0], dofs)


def test_nedelec_interpolation_energy_and_derivatives():
    """Compile interpolation, its energy and directional derivatives."""
    cell = basix_cell(basix.CellType.triangle)
    topology = static_topology((cell,), ((0, 1, 2),))
    space = basix_space("P2-vector", topology, basix.ElementFamily.P, 2, shape=(2,))
    u, h = function(space, "u"), function(space, "h")
    source = space.element(0)
    field = pull_back(u, cell_map(topology, 0))
    target = basix_space("Nedelec", cell, basix.ElementFamily.N1E, 1)
    reduced = interpolate(field, target, sampling_rule=edge_sampling_rule(target.element(0)))
    rng = np.random.default_rng(42)
    coefficients, direction = rng.normal(size=(2, source.dim, 2))
    env = {"coefficients:u": coefficients.ravel(), "coefficients:h": direction.ravel()}
    points = np.array([[0.13, 0.21], [0.6, 0.1], [0.0, 0.0], [0.2, 0.8]])
    compiled = [compile_form(point_value(reduced, p, cell), 4) for p in points]
    actual = np.array([evaluate(e, env) for e in compiled])
    expected = reference_values(source, coefficients, points)
    np.testing.assert_allclose(actual, expected, atol=1e-12)
    print(f"P2-vector -> Nedelec: max error {np.max(np.abs(actual - expected)):.3e}")

    # Every lowest-order Nedelec field must be reproduced exactly.
    p = source.points
    exact_coefficients = np.column_stack((2 - 3 * p[:, 1], -1 + 3 * p[:, 0]))
    actual = np.array(
        [evaluate(e, {"coefficients:u": exact_coefficients.ravel()}) for e in compiled]
    )
    expected = np.column_stack((2 - 3 * points[:, 1], -1 + 3 * points[:, 0]))
    np.testing.assert_allclose(actual, expected, atol=1e-12)
    print("Nedelec field reproduction: passed")

    q, weights = basix.make_quadrature(basix.CellType.triangle, 4)
    energy = integral(dot(reduced, reduced) / num(2), cell)
    derivative = directional_derivative(energy, u, u, h)
    second = directional_derivative(derivative, u, u, h)
    values = reference_values(source, coefficients, q)
    variations = reference_values(source, direction, q)
    expected = (
        np.einsum("q,qc,qc->", weights, values, values) / 2,
        np.einsum("q,qc,qc->", weights, values, variations),
        np.einsum("q,qc,qc->", weights, variations, variations),
    )
    for label, expression, wanted in zip(
        ("energy", "first derivative", "second derivative"),
        (energy, derivative, second),
        expected,
        strict=True,
    ):
        actual = evaluate(compile_form(expression, 4), env)
        np.testing.assert_allclose(actual, wanted, atol=1e-12)
        print(f"Compiled {label}: error {abs(actual - wanted):.3e}")
