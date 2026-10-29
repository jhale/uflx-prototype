import basix
import numpy as np
from test_basix import CELLS, square, table
from testutil import vertex_mesh

from uflx_oracle import evaluate as oracle
from uflx_spec_interpreter import (
    RD,
    component_tensor,
    coordinate,
    domain_map,
    expand_integrals,
    function,
    gradient,
    index,
    indexed,
    integral,
    num,
    pull_back,
)
from uflx_spec_interpreter._src.nodes import Gradient, Inverse, PullBack
from uflx_spec_interpreter._src.passes import gradients
from uflx_spec_interpreter._src.substitution import everywhere
from uflx_toy.basix_cells import basix_cell
from uflx_toy.basix_spaces import basix_space
from uflx_toy.topology import static_topology

K = basix_cell(CELLS[2])
D = square(K)[0]
X, k = coordinate(D), index("k")


def core(e):
    return gradients.lower(expand_integrals(e))


def nodes(e):
    found = []
    everywhere(e, lambda n: found.append(n) or True)
    return found


def test_a_gradient_pulled_back_by_a_map_that_is_not_a_cell_needs_no_inverse():
    # The gradient of an analytic field on RD(2) at the points of a deformed mesh.
    x = coordinate(RD(2))
    shift = num(0.1) * X[0] * X[1] * indexed(table([1.0, 2.0]), (k,))
    phi = domain_map(D, component_tensor(X[k] + shift, (k,)), RD(2))
    f = integral(pull_back(gradient(x[0] * x[0] * x[1], RD(2)), phi)[0], D)
    out = core(f)
    assert gradients.in_n6(out) and not any(isinstance(n, Inverse) for n in nodes(out))
    assert np.isclose(oracle(out, {})(()), oracle(f, {})(()))


def test_a_gradient_of_the_coordinates_of_a_mesh_needs_no_inverse():
    # Gradients on the mesh come before the pull-backs by its cells, so K never meets J.
    f = integral(gradient(X[0] * X[0] * X[1], D)[0], D)
    out = core(f)
    assert gradients.in_n6(out) and not any(isinstance(n, Inverse) for n in nodes(out))
    assert np.isclose(oracle(out, {})(()), oracle(f, {})(()))


def test_a_gradient_on_another_mesh_remains_for_the_points_of_the_mesh():
    # The other diagonal of the square, read through the coordinates of the first mesh.
    T2 = static_topology((K,), (((0, 1, 3), (1, 2, 3)),), ())
    D2 = vertex_mesh(T2, [(0, 0), (1, 0), (1, 1), (0, 1)])
    u = function(basix_space("U", D2, basix.ElementFamily.P, 1), "u")
    chi = domain_map(D, X, D2)
    out = core(integral(pull_back(gradient(u, D2), chi)[0], D))
    assert gradients.in_n6(out)
    # One map from the reference triangle of the first mesh into the second, for the location.
    leaves = [n for n in nodes(out) if isinstance(n, PullBack) and n.expression == Gradient(u, D2)]
    assert [(m.parameter_domain, m.ambient_domain) for m in (n.domain for n in leaves)] == [(K, D2)]
