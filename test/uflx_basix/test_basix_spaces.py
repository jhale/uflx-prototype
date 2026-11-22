import basix
import numpy as np
import pytest

from uflx._src.passes import integrals
from uflx._src.passes.contracts import in_core_normal_form
from uflx._src.passes.lowering import lower_spatial
from uflx.calculus import gradient, integral
from uflx.forms import function
from uflx.geometry import pull_back
from uflx.passes import expand_arithmetic
from uflx_basix.assembly import conforming_dofmap
from uflx_basix.basix_cells import basix_cell
from uflx_basix.basix_spaces import basix_space
from uflx_oracle import evaluate

from ..testutil import push_forward, side_map
from .test_basix import mixed, square

P = basix.ElementFamily.P
# The vertex coordinates of the square of test_basix and of its mixed mesh.
SQUARE = [(0, 0), (1, 0), (1, 1), (0, 1)]
MIXED = [(0, 0), (0.5, 0), (1, 0), (0, 1), (0.5, 1), (1, 1)]


def interpolate(V, f, coordinates):
    # The coefficients of the interpolant of f, at the points of the element in each cell.
    U = np.zeros(V.num_dofs)
    T = V.topology
    for j, vertices in enumerate(T.cells):
        E, cell_type = V.element(j), T.cell_types[j].cell_type
        shapes = basix.create_element(P, cell_type, 1).tabulate(0, E.points)[0, :, :, 0]
        x = shapes @ np.array([coordinates[v] for v in vertices], dtype=float)
        U[list(V.cell_dofs[j])] = [f(p) for p in x]
    return U


def value(e, env):
    out = lower_spatial(integrals.expand_integrals(e))
    out = expand_arithmetic(out)
    assert in_core_normal_form(out)
    before, after = evaluate(e, env)(()), evaluate(out, env)(())
    assert np.isclose(before, after, rtol=1e-6, atol=1e-9), (before, after)
    return np.real(after)


def jump_squared(u, D, E):
    plus, minus = side_map(D, E, "+"), side_map(D, E, "-")
    jump = pull_back(u, plus) - pull_back(u, minus)
    return integral(jump * jump, E)


@pytest.mark.parametrize("degree", [1, 2])
def test_an_interpolated_polynomial_integrates_exactly(degree):
    D, _, _ = square(basix_cell(basix.CellType.triangle))
    V = basix_space("V", D.topology, P, degree)
    uT = function(V, "u")
    u = push_forward(uT, D)
    f = (lambda x: 1 + x[0] + 2 * x[1]) if degree == 1 else (lambda x: x[0] ** 2 + x[0] * x[1])
    env = {uT: interpolate(V, f, SQUARE)}
    expected = 2.5 if degree == 1 else 1 / 3 + 1 / 4
    assert np.isclose(value(integral(u, D), env), expected)
    assert np.isclose(value(integral(gradient(u, D)[1], D), env), 2 if degree == 1 else 0.5)


def test_library_gathering_makes_traces_continuous():
    # The library gathers shared DOFs into independent local coefficient slots.
    D, E, _ = square(basix_cell(basix.CellType.triangle))
    rng = np.random.default_rng(3)
    for degree in (1, 2):
        V = basix_space("V", D.topology, P, degree)
        uT = function(V, "u")
        mapping = conforming_dofmap(V)
        coefficients = {uT: mapping.gather(rng.normal(size=mapping.num_dofs))}
        assert np.isclose(
            value(jump_squared(push_forward(uT, D), D, E), coefficients), 0, atol=1e-10
        )
    W = basix_space("W", D.topology, P, 1, discontinuous=True)
    wT = function(W, "w")
    w = push_forward(wT, D)
    assert value(jump_squared(w, D, E), {wT: rng.normal(size=W.num_dofs)}) > 1e-3
    # A discontinuous space that interpolates a continuous function has no jump either.
    env = {wT: interpolate(W, lambda x: 1 + x[0] - x[1], SQUARE)}
    assert np.isclose(value(jump_squared(w, D, E), env), 0, atol=1e-10)


def test_a_continuous_space_on_triangles_and_a_quadrilateral():
    D, E, _ = mixed()
    V = basix_space("V", D.topology, P, 2)
    uT = function(V, "u")
    u = push_forward(uT, D)
    env = {uT: interpolate(V, lambda x: x[0] ** 2 + x[0] * x[1] - x[1], MIXED)}
    assert np.isclose(value(integral(u, D), env), 1 / 3 + 1 / 4 - 1 / 2)
    rng = np.random.default_rng(4)
    mapping = conforming_dofmap(V)
    random = {uT: mapping.gather(rng.normal(size=mapping.num_dofs))}
    assert np.isclose(value(jump_squared(u, D, E), random), 0, atol=1e-10)
