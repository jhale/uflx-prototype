import basix
import numpy as np
from test_basix import square

from uflx_oracle import evaluate as oracle
from uflx_spec_interpreter import (
    RD,
    component_tensor,
    coordinate,
    domain_map,
    expand_integrals,
    gradient,
    index,
    integral,
    num,
    pull_back,
)
from uflx_spec_interpreter._src.nodes import Inverse
from uflx_spec_interpreter._src.passes import gradients
from uflx_spec_interpreter._src.substitution import everywhere
from uflx_toy.basix_cells import basix_cell

K = basix_cell(basix.CellType.triangle)
D = square(K)[0]
X, k = coordinate(D), index("k")


def core(e):
    return gradients.lower(expand_integrals(e))


def nodes(e):
    found = []
    everywhere(e, lambda n: found.append(n) or True)
    return found


def test_a_gradient_pulled_back_by_a_map_that_is_not_a_cell_needs_no_inverse():
    # A coordinate-domain map uses the chain rule without inverting its Jacobian.
    P = RD(2)
    x = coordinate(P)
    phi = domain_map(P, component_tensor(x[k] + num(0.1) * x[0] * x[1], (k,)), P)
    f = pull_back(gradient(x[0] * x[0] * x[1], P), phi)[0]
    out = core(f)
    assert gradients.in_n6(out) and not any(isinstance(n, Inverse) for n in nodes(out))
    from uflx_oracle import at

    assert np.isclose(
        at(P, np.array([0.2, 0.3]), lambda: oracle(out, {})(())),
        at(P, np.array([0.2, 0.3]), lambda: oracle(f, {})(())),
    )


def test_a_gradient_of_the_coordinates_of_a_mesh_needs_no_inverse():
    # Gradients on the mesh come before the pull-backs by its cells, so K never meets J.
    f = integral(gradient(X[0] * X[0] * X[1], D)[0], D)
    out = core(f)
    assert gradients.in_n6(out) and not any(isinstance(n, Inverse) for n in nodes(out))
    assert np.isclose(oracle(out, {})(()), oracle(f, {})(()))
