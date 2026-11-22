import basix
import numpy as np

from uflx._src.algorithms.traversal import everywhere
from uflx._src.nodes import Gradient, PushForward
from uflx._src.passes.lowering import lower_spatial
from uflx.arithmetic import num
from uflx.calculus import coordinate, curl, divergence, gradient, integral
from uflx.forms import function
from uflx.geometry import mesh
from uflx.nodes import EuclideanSpace
from uflx.passes import expand_integrals
from uflx.tensors import index, index_sum, tensor
from uflx_basix.basix_cells import basix_cell
from uflx_basix.basix_spaces import basix_space
from uflx_compiler import compile_form, evaluate
from uflx_oracle import evaluate as oracle

from ..testutil import grouped_topology, push_forward, vertex_mesh

P, RT, N1 = basix.ElementFamily.P, basix.ElementFamily.RT, basix.ElementFamily.N1E
K = basix_cell(basix.CellType.triangle)
# The vertices of each cell are in increasing order, so the facets of Basix need no signs.
T = grouped_topology((K,), (((0, 1, 2), (0, 2, 3)),), ())
VERTICES = [(0, 0), (1, 0), (1.1, 0.9), (-0.1, 1)]


def curved():
    # A P2 coordinate function with the shared diagonal bent.
    X = basix_space("X", T, P, 2)
    x0, x1 = function(X, "x0"), function(X, "x1")
    points = basix.create_element(
        P, basix.CellType.triangle, 2, basix.LagrangeVariant.equispaced
    ).points
    values = np.zeros((X.num_dofs, 2))
    for c, rows in enumerate(X.cell_dofs):
        a, b, d = np.asarray(VERTICES, dtype=float)[list(T.cells[c])]
        values[list(rows)] = a + np.outer(points[:, 0], b - a) + np.outer(points[:, 1], d - a)
    middle = np.all(np.isclose(values, (0.55, 0.45)), axis=1)
    values[middle] += (0.06, -0.04)
    geometry = {x0: values[:, 0], x1: values[:, 1]}
    return mesh(tensor((x0, x1)), EuclideanSpace(2)), geometry


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


def second_derivatives(e):
    # The gradients of gradients in the lowered form, such as the derivative of the Jacobian.
    found = []
    everywhere(
        lower_spatial(expand_integrals(e)),
        lambda n: (
            not (isinstance(n, Gradient) and isinstance(n.expression, Gradient))
            or not found.append(n)
        ),
    )
    return found


def test_the_divergence_of_a_raviart_thomas_function_is_the_reference_divergence():
    i = index("i")
    for D, geometry in ((vertex_mesh(T, VERTICES), {}), curved()):
        vT = function(basix_space("R", T, RT, 1), "v")
        v = push_forward(vT, D)
        W = np.random.default_rng(2).normal(size=vT.space.num_dofs)
        data = by_label(geometry) | {"coefficients:v": W}
        by_hand = index_sum(gradient(v, D)[i, i], (i,))
        for f, g in (
            (divergence(v, D), by_hand),
            (divergence(v, D) * divergence(v, D), by_hand * by_hand),
        ):
            value = evaluate(compile_form(integral(f, D), 12), data)
            assert np.isclose(value, evaluate(compile_form(integral(g, D), 12), data))
            assert np.isclose(value, oracle(integral(f, D), geometry | {vT: W})(()), rtol=1e-6)
        # The identity needs no derivative of the Jacobian, so no second derivative is read.
        assert not second_derivatives(integral(divergence(v, D), D))


def test_the_curl_of_a_nedelec_function_is_the_reference_curl():
    for D, geometry in ((vertex_mesh(T, VERTICES), {}), curved()):
        uT = function(basix_space("N", T, N1, 1), "u")
        u = push_forward(uT, D)
        W = np.random.default_rng(3).normal(size=uT.space.num_dofs)
        data = by_label(geometry) | {"coefficients:u": W}
        g = gradient(u, D)
        f, by_hand = (
            integral(curl(u, D) * curl(u, D), D),
            integral((g[1, 0] - g[0, 1]) * (g[1, 0] - g[0, 1]), D),
        )
        value = evaluate(compile_form(f, 12), data)
        assert np.isclose(value, evaluate(compile_form(by_hand, 12), data))
        assert np.isclose(value, oracle(f, geometry | {uT: W})(()), rtol=1e-6)
        assert not second_derivatives(f)


def test_the_curl_of_a_nedelec_function_on_a_tetrahedron_is_the_reference_curl():
    K3 = basix_cell(basix.CellType.tetrahedron)
    T3 = grouped_topology((K3,), (((0, 1, 2, 3),),), ())
    D = vertex_mesh(T3, [(0, 0, 0), (1, 0.1, 0), (0.2, 1, 0.1), (0.1, 0.2, 0.9)])
    uT = function(basix_space("N", T3, N1, 1), "u")
    u = push_forward(uT, D)
    W = np.random.default_rng(4).normal(size=uT.space.num_dofs)
    i, c, g = index("i"), curl(u, D), gradient(u, D)
    by_hand = tensor((g[2, 1] - g[1, 2], g[0, 2] - g[2, 0], g[1, 0] - g[0, 1]))
    f = integral(index_sum(c[i] * c[i], (i,)), D)
    value = evaluate(compile_form(f, 4), {"coefficients:u": W})
    expected = integral(index_sum(by_hand[i] * by_hand[i], (i,)), D)
    assert np.isclose(value, evaluate(compile_form(expected, 4), {"coefficients:u": W}))
    assert np.isclose(value, oracle(f, {uT: W})(()), rtol=1e-6)


def test_the_divergence_of_a_field_with_no_piola_map_is_the_trace_of_its_gradient():
    D, geometry = curved()
    data = by_label(geometry)
    trace = evaluate(compile_form(integral(divergence(coordinate(D), D), D), 4), data)
    assert np.isclose(trace, 2 * evaluate(compile_form(integral(num(1), D), 4), data))
