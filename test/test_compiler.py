import re

import basix
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from test_basix import mixed, quadrilaterals, square
from testutil import contains

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
from uflx_oracle import evaluate as oracle
from uflx_spec_interpreter import (
    RD,
    cell_domain,
    cell_index,
    coordinate,
    domain_map,
    form,
    function,
    gradient,
    index,
    index_sum,
    integral,
    normal,
    num,
    pull_back,
    side_map,
    tensor,
)
from uflx_spec_interpreter._draft import coordinate_function, image, push_forward
from uflx_spec_interpreter._src.nodes import (
    Coordinate,
    Division,
    Gradient,
    Integral,
    Inverse,
    Power,
    PullBack,
)
from uflx_spec_interpreter._src.substitution import everywhere, expressions, memoized
from uflx_toy.basix_cells import basix_cell
from uflx_toy.basix_spaces import basix_space
from uflx_toy.runtime import runtime_array, runtime_extent
from uflx_toy.topology import static_topology

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
    D = square(basix_cell(basix.CellType.triangle))[0] if mesh == "triangles" else mixed()[0]
    V = basix_space("V", D.topology, P, degree)
    u = push_forward(function(V, "u"), D)
    U = np.random.default_rng(degree).normal(size=V.num_dofs)
    for e in functionals(D, u):
        compiled = compile_form(e, 3 * degree + 2)
        # Quadrature and tabulation leave arithmetic on arrays, with no lazy node.
        lazy = (Integral, PullBack, Gradient, Coordinate)
        assert not contains(compiled, lambda n, lazy=lazy: isinstance(n, lazy))
        env = {"coefficients:u": U}
        expected = oracle(e, env)(())
        assert np.isclose(evaluate(compiled, env), expected, rtol=1e-8), e
        assert np.isclose(evaluate(compiled, env, np, fold(compiled)), expected, rtol=1e-8), e


def test_the_same_graph_runs_on_jax():
    D = mixed()[0]
    V = basix_space("V", D.topology, P, 2)
    u = push_forward(function(V, "u"), D)
    U = np.random.default_rng(5).normal(size=V.num_dofs)
    compiled = compile_form(functionals(D, u)[3], 8)
    key = "coefficients:u"
    f = jax.jit(lambda U: evaluate(compiled, {key: U}, jnp))
    assert np.isclose(f(jnp.asarray(U)), evaluate(compiled, {key: U}))
    # The derivative in the coefficients is the residual vector, here by automatic differentiation.
    d = np.random.default_rng(6).normal(size=V.num_dofs)
    step = 1e-6
    central = evaluate(compiled, {key: U + step * d}) - evaluate(compiled, {key: U - step * d})
    assert np.isclose(jax.grad(f)(jnp.asarray(U)) @ d, central / (2 * step), rtol=1e-6)


def instructions(compiled, key, n, folded=None):
    f = jax.jit(lambda U: evaluate(compiled, {key: U}, jnp, folded))
    return len(re.findall(r"= \S+ \w[\w-]*\(", f.lower(jnp.zeros(n)).compile().as_text()))


def test_folding_the_geometry_shrinks_the_kernel():
    # The geometry of a mesh with constant coordinates reads no kernel input, so it is folded.
    D = mixed()[0]
    V = basix_space("V", D.topology, P, 2)
    u = push_forward(function(V, "u"), D)
    compiled = compile_form(functionals(D, u)[3], 4)
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
        integral(pull_back(u, plus) * normal(plus)[0], E),
        integral(jump * jump, E),
        integral(pull_back(g[0], inside) * normal(inside)[0], B),
        integral(pull_back(u, minus) * pull_back(g[1], plus), E),
    ]


@pytest.mark.parametrize(
    "mesh",
    [lambda: square(basix_cell(basix.CellType.triangle)), quadrilaterals, mixed],
    ids=["triangles", "quadrilaterals", "mixed"],
)
@pytest.mark.parametrize("discontinuous", [False, True])
def test_compiled_facet_integrals_agree_with_the_oracle(mesh, discontinuous):
    D, E, B = mesh()
    V = basix_space("V", D.topology, P, 1, discontinuous)
    u = push_forward(function(V, "u"), D)
    U = np.random.default_rng(8).normal(size=V.num_dofs)
    env = {"coefficients:u": U}
    for e in facet_functionals(D, E, B, u):
        compiled = compile_form(e, 4)
        lazy = (Integral, PullBack, Gradient, Coordinate)
        assert not contains(compiled, lambda n, lazy=lazy: isinstance(n, lazy))
        expected = oracle(e, env)(())
        assert np.isclose(
            evaluate(compiled, env, np, fold(compiled)), expected, rtol=1e-8, atol=1e-12
        ), e


def test_a_function_at_points_known_only_at_assembly():
    # The cut cell example on a mesh: each cell has its own points, which the solver gives.
    D = square(basix_cell(basix.CellType.triangle))[0]
    V = basix_space("V", D.topology, P, 2)
    u = push_forward(function(V, "u"), D)
    K = D.pieces[0].parameter_domain
    ((c, cells),) = D.pieces[0].parameterisation.free_indices
    q, i = runtime_extent("q"), index("i")
    points = runtime_array("points", (cells, q, 2))
    weights = runtime_array("weights", (cells, q))
    xs = domain_map(RD(0), points[c, i], K)
    cell = cell_domain(D, 0)
    rng = np.random.default_rng(11)
    U = rng.normal(size=V.num_dofs)
    for f in (u, gradient(u, D)[0]):
        e = index_sum(
            index_sum(weights[c, i] * integral(pull_back(pull_back(f, cell), xs), RD(0)), (i,)),
            (c,),
        )
        compiled = compile_form(e, 2)
        for n in (1, 3, 6):
            # Points in the reference triangle, as a cut of each cell would give.
            X = rng.dirichlet(np.ones(3), size=(cells, n))[..., 1:]
            W = rng.random((cells, n))
            # The consumer passes the inputs that the kernel states, by label, to both.
            assert kernel_inputs(compiled) == {
                "points": (2, "q", 2),
                "weights": (2, "q"),
                "coefficients:u": (V.num_dofs,),
            }
            given = {"q": n, "points": X, "weights": W, "coefficients:u": U}
            env = runtime_tables(compiled, given)
            expected = oracle(e, given)(())
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
    T = static_topology((K,), (((0, 1, 2), (0, 2, 3)),), ())
    X = basix_space("X", T, P, 1, discontinuous=True)
    x0, x1 = function(X, "x0"), function(X, "x1")

    def geometry(cells):
        cells = np.asarray(cells, dtype=float)
        return {"coefficients:x0": cells[..., 0].ravel(), "coefficients:x1": cells[..., 1].ravel()}

    return geometry, cell_index(T, 0), image(tensor((x0, x1)), RD(2))


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
    expected = oracle(energy, env)(())
    assert np.isclose(evaluate(compiled, env, np, fold(compiled)), expected, rtol=1e-8)
    M = assemble(form(integral(u * v, D), (vT, uT)), 4)(geometry(coordinates))
    expected = oracle(integral(u * v, D), geometry(coordinates) | {uT: U, vT: W})(())
    assert np.isclose(W @ M @ U, expected)


def test_the_geometry_of_an_affine_mesh_is_one_node_for_each_cell_in_every_form():
    _, c, D = runtime_mesh()
    V = basix_space("V", D.topology, P, 2)
    u, w = push_forward(function(V, "u"), D), push_forward(function(V, "w"), D)
    gu, gw = gradient(u, D), gradient(w, D)
    forms = [integral(gu[0] * gu[0], D), integral(gu[1] * gw[0], D)]
    found = [set() for _ in forms]
    for dets, f in zip(found, forms, strict=True):
        everywhere(
            compile_form(f, 4),
            lambda n: not isinstance(n, Division) or not dets.add(n.second),
        )
    # K is the adjugate over det J, and no zero term adds the index of the points to det J.
    assert found[0] == found[1] and len(found[0]) == 1
    assert [i for i, _ in next(iter(found[0])).free_indices] == [c]


def mesh_only(e, D):
    # The largest subexpressions that read the coordinates and no other input.
    geometry = {coefficients(f) for f in coordinate_function(D).entries}
    reads, found = memoized(arrays), set()

    def walk(walk, n):
        if reads(n) and set(reads(n)) <= geometry:
            found.add(n)
        else:
            for m in expressions(n):
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
    assert np.isclose(evaluate(compiled, env), oracle(f, env)(()))
