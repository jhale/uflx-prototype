import re

import basix
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from uflx._src.algorithms.traversal import everywhere, memoized
from uflx._src.meshes import cell_map
from uflx._src.nodes import Coordinate, Division, Gradient, Integral, Inverse, Power, PullBack
from uflx._src.nodes.structure import expression_operands
from uflx.algorithms import replace
from uflx.arithmetic import num
from uflx.calculus import coordinate, gradient, integral, reference_integral
from uflx.forms import form, function
from uflx.geometry import domain_map, mesh, pull_back
from uflx.nodes import EuclideanSpace
from uflx.tensors import index, index_sum, tensor
from uflx_basix.basix_cells import basix_cell
from uflx_basix.basix_spaces import basix_space
from uflx_basix.runtime import runtime_array, runtime_extent
from uflx_basix.sides import outward_conormal
from uflx_compiler import (
    assemble,
    coefficients,
    compile_form,
    evaluate,
    fold,
    kernel_inputs,
    runtime_tables,
)
from uflx_compiler.inputs import arrays
from uflx_compiler.quadrature import constant
from uflx_oracle.interpreter import Interpreter

from ..testutil import contains
from ..uflx_basix.meshes import (
    grouped_topology,
    mixed,
    push_forward,
    quadrilaterals,
    side_map,
    square,
)

jax.config.update("jax_enable_x64", True)
P = basix.ElementFamily.P


def functionals(D, u):
    g = gradient(u, D)
    return [
        integral(num(1), D),
        integral(u, D),
        integral(u * u, D),
        integral(g[0] * g[0] + g[1] * g[1], D),
        integral(u * g[0] * g[1], D),
    ]


@pytest.mark.parametrize(
    ("mesh", "degree"),
    [("triangles", 1), ("triangles", 2), ("mixed", 1), ("mixed", 2)],
)
def test_compiled_functionals_agree_with_the_oracle(mesh, degree):
    D, _, _, geometry = (
        square(basix_cell(basix.CellType.triangle)) if mesh == "triangles" else mixed()
    )
    V = basix_space("V", D.topology, P, degree)
    u = push_forward(function(V, "u"), D)
    U = np.random.default_rng(degree).normal(size=V.num_dofs)
    for e in functionals(D, u):
        compiled = compile_form(e, 3 * degree + 2)
        # Quadrature and tabulation leave arithmetic on arrays, with no lazy node.
        lazy = (Integral, PullBack, Gradient, Coordinate)
        assert not contains(compiled, lambda n, lazy=lazy: isinstance(n, lazy))
        env = geometry | {"coefficients:u": U}
        expected = Interpreter(env).evaluate(e)
        assert np.isclose(evaluate(compiled, env), expected, rtol=1e-8), e
        assert np.isclose(evaluate(compiled, env, np, fold(compiled)), expected, rtol=1e-8), e


def test_tensor_quadrature_over_mixed_cells_preserves_free_indices():
    D, _, _, geometry = mixed()
    x, y = coordinate(D)[0], coordinate(D)[1]
    i = index("load")
    scale = tensor((num(1), num(3)))[i]
    f = scale * tensor((tensor((num(1), x)), tensor((y, x * y))))

    whole = compile_form(integral(f, D), 2)
    pieces = [
        compile_form(reference_integral(f, D, j), 2) for j in range(len(D.topology.cell_types))
    ]
    for result in [whole, *pieces]:
        assert result.value_shape == (2, 2)
        assert result.free_indices == ((i, 2),)

    # Integrals of 1, x, y and xy over the unit square, for each load.
    expected = np.array([1, 3])[:, None, None] * np.array([[1, 0.5], [0.5, 0.25]])
    np.testing.assert_allclose(evaluate(whole, geometry), expected)
    np.testing.assert_allclose(sum(evaluate(piece, geometry) for piece in pieces), expected)


def test_the_same_graph_runs_on_jax():
    D, _, _, geometry = mixed()
    V = basix_space("V", D.topology, P, 2)
    u = push_forward(function(V, "u"), D)
    U = np.random.default_rng(5).normal(size=V.num_dofs)
    compiled = compile_form(functionals(D, u)[3], 8)
    key = "coefficients:u"
    f = jax.jit(lambda U: evaluate(compiled, geometry | {key: U}, jnp))
    assert np.isclose(f(jnp.asarray(U)), evaluate(compiled, geometry | {key: U}))
    # The derivative in the coefficients is the residual vector, here by automatic differentiation.
    d = np.random.default_rng(6).normal(size=V.num_dofs)
    step = 1e-6
    central = evaluate(compiled, geometry | {key: U + step * d}) - evaluate(
        compiled, geometry | {key: U - step * d}
    )
    assert np.isclose(jax.grad(f)(jnp.asarray(U)) @ d, central / (2 * step), rtol=1e-6)


def instructions(compiled, key, n, folded=None):
    f = jax.jit(lambda U: evaluate(compiled, {key: U}, jnp, folded))
    return len(re.findall(r"= \S+ \w[\w-]*\(", f.lower(jnp.zeros(n)).compile().as_text()))


def test_folding_the_geometry_shrinks_the_kernel():
    # The geometry of a mesh with constant coordinates reads no kernel input, so it is folded.
    D, _, _, geometry = mixed()
    V = basix_space("V", D.topology, P, 2)
    u = push_forward(function(V, "u"), D)
    compiled = compile_form(functionals(D, u)[3], 4)
    # Specialise the coordinate coefficient to its known values before folding.
    compiled = replace(
        compiled, {coefficients(D.coordinate_function): constant(geometry["coefficients:x"])}
    )
    key = "coefficients:u"
    folded = fold(compiled)
    assert compiled not in folded and folded
    plain, small = (
        instructions(compiled, key, V.num_dofs),
        instructions(compiled, key, V.num_dofs, folded),
    )
    # The simplification already removes most of the work that folding would.
    assert small < plain / 1.5, (plain, small)
    U = np.random.default_rng(7).normal(size=V.num_dofs)
    f = jax.jit(lambda U: evaluate(compiled, {key: U}, jnp, folded))
    assert np.isclose(f(jnp.asarray(U)), evaluate(compiled, {key: U}))


def facet_functionals(D, E, B, u):
    plus, minus, inside = side_map(D, E, "+"), side_map(D, E, "-"), side_map(D, B, "inside")
    jump = pull_back(u, plus) - pull_back(u, minus)
    g = gradient(u, D)
    return [
        integral(pull_back(u, plus) * outward_conormal(plus)[0], E),
        integral(jump * jump, E),
        integral(pull_back(g[0], inside) * outward_conormal(inside)[0], B),
        integral(pull_back(u, minus) * pull_back(g[1], plus), E),
    ]


@pytest.mark.parametrize(
    "mesh",
    [lambda: square(basix_cell(basix.CellType.triangle)), quadrilaterals, mixed],
    ids=["triangles", "quadrilaterals", "mixed"],
)
@pytest.mark.parametrize("discontinuous", [False, True])
def test_compiled_facet_integrals_agree_with_the_oracle(mesh, discontinuous):
    D, E, B, geometry = mesh()
    V = basix_space("V", D.topology, P, 1, discontinuous)
    u = push_forward(function(V, "u"), D)
    U = np.random.default_rng(8).normal(size=V.num_dofs)
    env = geometry | {"coefficients:u": U}
    for e in facet_functionals(D, E, B, u):
        compiled = compile_form(e, 4)
        lazy = (Integral, PullBack, Gradient, Coordinate)
        assert not contains(compiled, lambda n, lazy=lazy: isinstance(n, lazy))
        expected = Interpreter(env).evaluate(e)
        assert np.isclose(
            evaluate(compiled, env, np, fold(compiled)), expected, rtol=1e-8, atol=1e-12
        ), e


def test_a_function_at_points_known_only_at_assembly():
    # The cut cell example on a mesh: each cell has its own points, which the solver gives.
    D, _, _, geometry = square(basix_cell(basix.CellType.triangle))
    V = basix_space("V", D.topology, P, 2)
    u = push_forward(function(V, "u"), D)
    K = D[0].reference_cell
    cells = len(D.topology.cell_types)
    q, i = runtime_extent("q"), index("i")
    points = runtime_array("points", (cells, q, 2))
    weights = runtime_array("weights", (cells, q))
    rng = np.random.default_rng(11)
    U = rng.normal(size=V.num_dofs)
    for f in (u, gradient(u, D)[0]):
        e = sum(
            (
                index_sum(
                    weights[slot, i]
                    * integral(
                        pull_back(
                            pull_back(f, cell_map(D, slot)),
                            domain_map(EuclideanSpace(0), points[slot, i], K),
                        ),
                        EuclideanSpace(0),
                    ),
                    (i,),
                )
                for slot in range(cells)
            ),
            num(0),
        )
        compiled = compile_form(e, 2)
        for n in (1, 3, 6):
            # Points in the reference triangle, as a cut of each cell would give.
            X = rng.dirichlet(np.ones(3), size=(cells, n))[..., 1:]
            W = rng.random((cells, n))
            # The consumer passes the inputs that the kernel states, by label, to both.
            assert kernel_inputs(compiled) == (
                {key: v.shape for key, v in geometry.items()} if f is not u else {}
            ) | {
                "points": (2, "q", 2),
                "weights": (2, "q"),
                "coefficients:u": (V.num_dofs,),
            }
            given = geometry | {"q": n, "points": X, "weights": W, "coefficients:u": U}
            env = runtime_tables(compiled, given)
            expected = Interpreter(given).evaluate(e)
            assert np.isclose(evaluate(compiled, env), expected, rtol=1e-8)
            # JAX takes the arrays in a fixed order, and the extent stays a Python int.
            keys = [k for k in env if k != "q"]

            def run(arrays, keys=keys, n=n, compiled=compiled):
                return evaluate(compiled, dict(zip(keys, arrays, strict=True)) | {"q": n}, jnp)

            arrays = tuple(jnp.asarray(env[k]) for k in keys)
            assert np.isclose(jax.jit(run)(arrays), expected)


def runtime_mesh():
    # A solver gives the coordinates of the vertices of each cell to the kernel, as for any mesh,
    # as the dofs of a discontinuous coordinate function of degree 1.
    K = basix_cell(basix.CellType.triangle)
    T = grouped_topology((K,), (((0, 1, 2), (0, 2, 3)),), ())
    X = basix_space("X", T, P, 1, discontinuous=True)
    x0, x1 = function(X, "x0"), function(X, "x1")

    def geometry(cells):
        cells = np.asarray(cells, dtype=float)
        return {"coefficients:x0": cells[..., 0].ravel(), "coefficients:x1": cells[..., 1].ravel()}

    return geometry, None, mesh(tensor((x0, x1)), EuclideanSpace(2))


def test_geometry_from_coordinates_given_at_assembly():
    geometry, _, D = runtime_mesh()
    V = basix_space("V", D.topology, P, 2)
    uT, vT = function(V, "u"), function(V, "v")
    u, v = push_forward(uT, D), push_forward(vT, D)
    rng = np.random.default_rng(12)
    # The square with its vertices moved, so the cells are not right triangles.
    corners = np.array([(0, 0), (1, 0), (1.2, 1.1), (-0.1, 0.9)]) + rng.normal(size=(4, 2)) * 0.05
    coordinates = corners[[[0, 1, 2], [0, 2, 3]]]
    U, W = rng.normal(size=V.num_dofs), rng.normal(size=V.num_dofs)
    g = gradient(u, D)
    energy = integral(g[0] * g[0] + g[1] * g[1], D)
    compiled = compile_form(energy, 4)
    env = geometry(coordinates) | {"coefficients:u": U}
    expected = Interpreter(env).evaluate(energy)
    assert np.isclose(evaluate(compiled, env, np, fold(compiled)), expected, rtol=1e-8)
    M = assemble(form(integral(u * v, D), (vT, uT)), 4)(geometry(coordinates))
    expected = Interpreter(geometry(coordinates) | {uT: U, vT: W}).evaluate(integral(u * v, D))
    assert np.isclose(W @ M @ U, expected)


def test_the_geometry_of_an_affine_mesh_is_one_node_for_each_cell_in_every_form():
    _, _c, D = runtime_mesh()
    V = basix_space("V", D.topology, P, 2)
    u, w = push_forward(function(V, "u"), D), push_forward(function(V, "w"), D)
    gu, gw = gradient(u, D), gradient(w, D)
    forms = [integral(gu[0] * gu[0], D), integral(gu[1] * gw[0], D)]
    found = [set() for _ in forms]
    for dets, f in zip(found, forms, strict=True):
        everywhere(
            compile_form(f, 4),
            lambda n: not isinstance(n, Division) or not dets.add(n.denominator),
        )
    # K is the adjugate over det J, and no zero term adds the index of the points to det J.
    assert found[0] == found[1] and len(found[0]) == 2
    assert all(d.free_indices == () for d in found[0])


def mesh_only(e, D):
    # Subexpressions that read only coordinates, independent of product association.
    geometry = {coefficients(f) for f in D.coordinate_function.entries}
    reads, found = memoized(arrays), set()

    def walk(walk, n):
        if reads(n) and set(reads(n)) <= geometry:
            found.add(n)
        for m in expression_operands(n):
            walk(m)

    memoized(walk)(e)
    return found


def test_the_geometry_at_the_points_of_a_rule_is_one_node_in_every_form():
    _, _, D = runtime_mesh()
    V = basix_space("V", D.topology, P, 2)
    u, w = push_forward(function(V, "u"), D), push_forward(function(V, "w"), D)
    X = coordinate(D)
    parts = [mesh_only(compile_form(integral(X[0] * f, D), 4), D) for f in (u, w)]
    at_points = {n for n in parts[0] & parts[1] if quadrature_indices(n)}
    assert at_points


def quadrature_indices(n):
    return [i for i, _ in n.free_indices if i.label[0] == "compiler.quadrature"]


def test_the_simplification_reduces_the_gradient_of_a_coordinate_to_arithmetic():
    # The gradient of x0 squared is 2 x0 times a row of the identity, read at one component.
    geometry, _, D = runtime_mesh()
    u = push_forward(function(basix_space("V", D.topology, P, 2), "u"), D)
    X, g = coordinate(D), gradient(u, D)
    f = integral(gradient(X[0] ** num(2), D)[0] * g[0] + gradient(X[0] ** num(2), D)[1] * g[1], D)
    compiled = compile_form(f, 4)
    assert not contains(compiled, lambda n: isinstance(n, Power | Inverse))
    rng = np.random.default_rng(3)
    coordinates = np.array(
        [[(0.1, 0.0), (1.0, 0.2), (0.3, 0.9)], [(0.1, 0.0), (0.3, 0.9), (-0.2, 0.7)]]
    )
    U = rng.normal(size=u.expression.space.num_dofs)
    env = geometry(coordinates) | {"coefficients:u": U}
    assert np.isclose(evaluate(compiled, env), Interpreter(env).evaluate(f))
