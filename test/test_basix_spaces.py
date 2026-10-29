import basix
import numpy as np
import pytest
from test_basix import CELLS, mixed, square

from uflx_oracle import evaluate
from uflx_spec_interpreter import function, gradient, integral, pull_back, side_map
from uflx_spec_interpreter._src.passes import gradients, integrals
from uflx_toy.basix_cells import basix_cell
from uflx_toy.basix_spaces import basix_space

P = basix.ElementFamily.P
# The vertex coordinates of the square of test_basix and of its mixed mesh.
SQUARE = [(0, 0), (1, 0), (1, 1), (0, 1)]
MIXED = [(0, 0), (0.5, 0), (1, 0), (0, 1), (0.5, 1), (1, 1)]


def interpolate(V, f, coordinates):
    # The coefficients of the interpolant of f, at the points of the element in each cell.
    U = np.zeros(V.num_dofs)
    T = V.domain.topology
    for j, cells in enumerate(T.cells):
        E, cell_type = V.element(j), T.cell_types[j].cell_type
        shapes = basix.create_element(P, cell_type, 1).tabulate(0, E.points)[0, :, :, 0]
        for c, vertices in enumerate(cells):
            x = shapes @ np.array([coordinates[v] for v in vertices], dtype=float)
            U[list(V.dofmap[j][c])] = [f(p) for p in x]
    return U


def value(e, env):
    out = gradients.lower(integrals.expand_integrals(e))
    assert gradients.in_n6(out)
    before, after = evaluate(e, env)(()), evaluate(out, env)(())
    assert np.isclose(before, after, rtol=1e-6, atol=1e-9), (before, after)
    return np.real(after)


def jump_squared(u, D, E):
    plus, minus = side_map(D, E, "+"), side_map(D, E, "-")
    jump = pull_back(u, plus) - pull_back(u, minus)
    return integral(jump * jump, E)


@pytest.mark.parametrize("degree", [1, 2])
def test_an_interpolated_polynomial_integrates_exactly(degree):
    D, _, _ = square(basix_cell(CELLS[2]))
    V = basix_space("V", D, P, degree)
    u = function(V, "u")
    f = (lambda x: 1 + x[0] + 2 * x[1]) if degree == 1 else (lambda x: x[0] ** 2 + x[0] * x[1])
    env = {u: interpolate(V, f, SQUARE)}
    expected = 2.5 if degree == 1 else 1 / 3 + 1 / 4
    assert np.isclose(value(integral(u, D), env), expected)
    assert np.isclose(value(integral(gradient(u, D)[1], D), env), 2 if degree == 1 else 0.5)


def test_a_continuous_space_has_no_jump_for_any_coefficients():
    # The two cells share the dofs of the diagonal, whatever the values of the dofs.
    D, E, _ = square(basix_cell(CELLS[2]))
    rng = np.random.default_rng(3)
    for degree in (1, 2):
        V = basix_space("V", D, P, degree)
        u = function(V, "u")
        coefficients = {u: rng.normal(size=V.num_dofs)}
        assert np.isclose(value(jump_squared(u, D, E), coefficients), 0, atol=1e-10)
    W = basix_space("W", D, P, 1, discontinuous=True)
    w = function(W, "w")
    assert value(jump_squared(w, D, E), {w: rng.normal(size=W.num_dofs)}) > 1e-3
    # A discontinuous space that interpolates a continuous function has no jump either.
    env = {w: interpolate(W, lambda x: 1 + x[0] - x[1], SQUARE)}
    assert np.isclose(value(jump_squared(w, D, E), env), 0, atol=1e-10)


def test_a_continuous_space_on_triangles_and_a_quadrilateral():
    D, E, _ = mixed()
    V = basix_space("V", D, P, 2)
    u = function(V, "u")
    env = {u: interpolate(V, lambda x: x[0] ** 2 + x[0] * x[1] - x[1], MIXED)}
    assert np.isclose(value(integral(u, D), env), 1 / 3 + 1 / 4 - 1 / 2)
    rng = np.random.default_rng(4)
    random = {u: rng.normal(size=V.num_dofs)}
    assert np.isclose(value(jump_squared(u, D, E), random), 0, atol=1e-10)
