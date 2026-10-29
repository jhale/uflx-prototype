import basix
import numpy as np
from test_basix import CELLS
from testutil import vertex_mesh

from uflx_compiler import compile_form, evaluate
from uflx_oracle import evaluate as oracle
from uflx_spec_interpreter import RD, function, gradient, index, index_sum, integral, tensor
from uflx_spec_interpreter._draft import image, push_forward
from uflx_spec_interpreter._src.nodes import PushForward
from uflx_spec_interpreter._src.substitution import everywhere
from uflx_toy.basix_cells import basix_cell
from uflx_toy.basix_spaces import basix_space
from uflx_toy.topology import static_topology

P, RT = basix.ElementFamily.P, basix.ElementFamily.RT
K = basix_cell(CELLS[2])
# The vertices of each cell are in increasing order, so the facets of Basix need no signs.
T = static_topology((K,), (((0, 1, 2), (0, 2, 3)),), ())
VERTICES = [(0, 0), (1, 0), (1.1, 0.9), (-0.1, 1)]


def curved():
    # A P2 coordinate function with the shared diagonal bent.
    X = basix_space("X", T, P, 2)
    x0, x1 = function(X, "x0"), function(X, "x1")
    points = basix.create_element(P, CELLS[2], 2, basix.LagrangeVariant.equispaced).points
    values = np.zeros((X.num_dofs, 2))
    for c, rows in enumerate(X.dofmap[0]):
        a, b, d = np.asarray(VERTICES, dtype=float)[list(T.cells[0][c])]
        values[list(rows)] = a + np.outer(points[:, 0], b - a) + np.outer(points[:, 1], d - a)
    middle = next(d for d in X.dofmap[0][0] if np.allclose(values[d], (0.55, 0.45)))
    values[middle] += (0.06, -0.04)
    geometry = {x0: values[:, 0], x1: values[:, 1]}
    return image(tensor((x0, x1)), RD(2)), geometry


def by_label(env):
    return {f"coefficients:{f.label}": v for f, v in env.items()}


def lowered(e):
    found = []
    everywhere(compile_form(e, 2), lambda n: not isinstance(n, PushForward) or not found.append(n))
    return not found


def raviart_thomas_forms(D, v):
    i = index("i")
    g = gradient(v, D)
    div = index_sum(g[i, i], (i,))
    return [
        integral(v[0] * v[0] + v[1] * v[1], D),
        integral(div, D),
        integral(div * div, D),
        integral(v[0] * g[1, 0] - v[1] * g[0, 1], D),
    ]


def test_a_raviart_thomas_function_pushed_forward_agrees_with_the_oracle():
    fluxes = []
    for D, geometry in ((vertex_mesh(T, VERTICES), {}), curved()):
        vT = function(basix_space("R", T, RT, 1), "v")
        v = push_forward(vT, D)
        assert v.value_shape == (2,) and v.law == basix_space("R", T, RT, 1).law
        W = np.random.default_rng(1).normal(size=vT.space.num_dofs)
        forms = raviart_thomas_forms(D, v)
        values = []
        for f in forms:
            values.append(evaluate(compile_form(f, 12), by_label(geometry) | {"coefficients:v": W}))
            # The oracle differentiates by differences, to about 1e-8.
            assert np.isclose(values[-1], oracle(f, geometry | {vT: W})(()), rtol=1e-6, atol=1e-6)
        fluxes.append(values[1])
    # Bending the shared diagonal leaves the boundary, so the divergence has the same integral.
    assert np.isclose(fluxes[0], fluxes[1], rtol=1e-12)
