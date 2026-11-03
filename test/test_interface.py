import basix
import numpy as np
import pytest
from testutil import push_forward, vertex_mesh

from uflx_compiler import assemble, compile_form, evaluate, runtime_tables
from uflx_oracle import evaluate as oracle
from uflx_spec_interpreter import (
    form,
    function,
    gradient,
    integral,
    pull_back,
    side_domain,
    side_map,
)
from uflx_toy.basix_cells import basix_cell
from uflx_toy.basix_spaces import basix_space
from uflx_toy.topology import static_topology

P = basix.ElementFamily.P
K, I = basix_cell(basix.CellType.triangle), basix_cell(basix.CellType.interval)
VERTICES = [(0, 0), (1, 0), (1, 1), (0, 1)]
TRIANGLES, INTERVALS = ((0, 1, 2), (0, 2, 3)), ((0, 1), (1, 2))


# Triangles, and a submesh of intervals on the bottom and right of the boundary.
T = static_topology((K,), (TRIANGLES,), ())
M = static_topology((I,), (INTERVALS,), ())
D, DM = vertex_mesh(T, VERTICES), vertex_mesh(M, VERTICES)
S = static_topology((I,), (INTERVALS,), ((("mesh", T, 0), ("submesh", M, 0)),))
E = side_domain(D, S, S.sides[0][0][0])
V, L = basix_space("V", T, P, 2), basix_space("L", M, P, 1, discontinuous=True)


def on(f, domain, side):
    return pull_back(f, side_map(domain, E, side))


def coupling(u, lam):
    return integral(on(lam, DM, "submesh") * on(u, D, "mesh"), E)


@pytest.mark.parametrize(
    "integrand",
    [
        lambda u, lam: on(lam, DM, "submesh") * on(u, D, "mesh"),
        lambda u, lam: on(lam, DM, "submesh") * on(gradient(u, D)[0], D, "mesh"),
        lambda u, lam: on(gradient(lam, DM)[0], DM, "submesh") * on(u, D, "mesh"),
    ],
)
def test_a_function_on_a_submesh_couples_to_a_function_on_the_mesh(integrand):
    u, lam = push_forward(function(V, "u"), D), push_forward(function(L, "lam"), DM)
    f = integral(integrand(u, lam), E)
    rng = np.random.default_rng(0)
    given = {
        "coefficients:u": rng.normal(size=V.num_dofs),
        "coefficients:lam": rng.normal(size=L.num_dofs),
    }
    compiled = compile_form(f, 3)
    assert np.isclose(evaluate(compiled, runtime_tables(compiled, given)), oracle(f, given)(()))


def test_the_block_of_a_lagrange_multiplier_on_a_submesh():
    uT, lamT = function(V, "u"), function(L, "lam")
    B = assemble(form(coupling(push_forward(uT, D), push_forward(lamT, DM)), (lamT, uT)), 3)({})
    assert B.shape == (L.num_dofs, V.num_dofs)
    rng = np.random.default_rng(1)
    U, Lam = rng.normal(size=V.num_dofs), rng.normal(size=L.num_dofs)
    w, z = function(V, "w"), function(L, "z")
    expected = oracle(coupling(push_forward(w, D), push_forward(z, DM)), {w: U, z: Lam})(())
    assert np.isclose(Lam @ B @ U, expected)
