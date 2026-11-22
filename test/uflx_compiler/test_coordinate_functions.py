import basix
import numpy as np

from uflx._src.algorithms.traversal import everywhere
from uflx._src.nodes import CellMap, Division, Gradient, PullBack
from uflx._src.passes.contracts import in_core_normal_form
from uflx._src.passes.lowering import lower_spatial
from uflx.arithmetic import num
from uflx.calculus import integral
from uflx.forms import function
from uflx.geometry import cell_map, mesh, pull_back
from uflx.nodes import EuclideanSpace
from uflx.passes import expand_integrals
from uflx_basix.basix_cells import basix_cell
from uflx_basix.basix_spaces import basix_space
from uflx_compiler import compile_form, evaluate
from uflx_oracle.interpreter import Interpreter

from ..uflx_basix.meshes import grouped_topology, push_forward, side_map, square
from ..uflx_basix.test_coordinate_functions import VERTICES, K, P, T, coordinates, p1_mesh
from .test_compiler import functionals, runtime_mesh


def labelled(env):
    return {f"coefficients:{f.label}": v for f, v in env.items()}


def nodes(e):
    found = []
    everywhere(e, lambda n: found.append(n) or True)
    return found


def test_scalar_and_blocked_p1_coordinate_fields_agree():
    D, geometry = p1_mesh()
    E, _, _, blocked_geometry = square(K)
    U = np.random.default_rng(0).normal(size=basix_space("V", T, P, 2).num_dofs)
    u, w = (
        push_forward(function(basix_space("V", T, P, 2), "u"), D),
        push_forward(function(basix_space("V", E.topology, P, 2), "u"), E),
    )
    for f, g in zip(functionals(D, u), functionals(E, w), strict=True):
        compiled = compile_form(f, 4)
        expected = evaluate(compile_form(g, 4), blocked_geometry | {"coefficients:u": U})
        assert np.isclose(evaluate(compiled, labelled(geometry) | {"coefficients:u": U}), expected)
        assert np.isclose(Interpreter(geometry | {u.expression: U}).evaluate(f), expected)


def test_the_jacobian_of_a_mesh_is_the_reference_gradient_of_the_coordinate_function():
    D, _ = p1_mesh()
    u = push_forward(function(basix_space("V", T, P, 2), "u"), D)
    out = lower_spatial(expand_integrals(functionals(D, u)[3]))
    assert in_core_normal_form(out)
    # The Jacobian is the reference gradient of the reads of x0 and x1 through the cell maps.
    gradients_of_reads = [
        n.expression.expression.label
        for n in nodes(out)
        if isinstance(n, Gradient)
        and isinstance(n.expression, PullBack)
        and isinstance(n.expression.map, CellMap)
    ]
    assert {"x0", "x1"} <= set(gradients_of_reads)


def test_a_curved_p2_coordinate_function_agrees_with_the_oracle():
    X, x0, x1, x = coordinates(2)
    D = mesh(x, EuclideanSpace(2))
    # Interpolate the square, then bend the shared diagonal.
    points = basix.create_element(
        P, basix.CellType.triangle, 2, basix.LagrangeVariant.equispaced
    ).points
    values = np.zeros((X.num_dofs, 2))
    for c, rows in enumerate(X.cell_dofs):
        a, b, d = VERTICES[list(T.cells[c])]
        values[list(rows)] = a + np.outer(points[:, 0], b - a) + np.outer(points[:, 1], d - a)
    midpoints = np.all(np.isclose(values, (0.5, 0.5)), axis=1)
    values[midpoints] += (0.08, -0.05)
    geometry = {x0: values[:, 0], x1: values[:, 1]}
    u = push_forward(function(basix_space("V", T, P, 2), "u"), D)
    U = np.random.default_rng(1).normal(size=u.expression.space.num_dofs)
    # The integrands are rational on a curved cell, so the rule has a high degree.
    for f in functionals(D, u):
        compiled = compile_form(f, 16)
        value = evaluate(compiled, labelled(geometry) | {"coefficients:u": U})
        assert np.isclose(
            value, Interpreter(geometry | {u.expression: U}).evaluate(f), rtol=1e-9
        ), f


def test_a_discontinuous_coordinate_function_from_the_vertices_of_each_cell():
    # The coordinates of the vertices of each cell are the dofs of a discontinuous P1 space.
    X, _, _, x = coordinates(1, discontinuous=True)
    D = mesh(x, EuclideanSpace(2))
    geometry_of, _, E = runtime_mesh()
    rng = np.random.default_rng(12)
    corners = np.array([(0, 0), (1, 0), (1.2, 1.1), (-0.1, 0.9)]) + rng.normal(size=(4, 2)) * 0.05
    cells = corners[[[0, 1, 2], [0, 2, 3]]]
    flat = cells.reshape(-1, 2)[np.argsort(np.ravel(X.cell_dofs))]
    geometry = {"coefficients:x0": flat[:, 0], "coefficients:x1": flat[:, 1]}
    U = rng.normal(size=basix_space("V", T, P, 2).num_dofs)
    u, w = (
        push_forward(function(basix_space("V", T, P, 2), "u"), D),
        push_forward(function(basix_space("V", E.topology, P, 2), "u"), E),
    )
    for f, g in zip(functionals(D, u), functionals(E, w), strict=True):
        value = evaluate(compile_form(f, 4), geometry | {"coefficients:u": U})
        expected = evaluate(compile_form(g, 4), geometry_of(cells) | {"coefficients:u": U})
        assert np.isclose(value, expected)


def test_fixed_cells_read_independent_fields():
    D, geometry = p1_mesh()
    u = push_forward(function(basix_space("V", T, P, 2), "u"), D)
    U = np.random.default_rng(2).normal(size=u.expression.space.num_dofs)
    reads = [integral(pull_back(u, cell_map(D, slot)), K) for slot in (1, 0)]
    env = labelled(geometry) | {"coefficients:u": U}
    for expression in reads:
        assert expression.free_indices == ()
        value = evaluate(compile_form(expression, 4), env)
        assert np.isclose(value, Interpreter(geometry | {u.expression: U}).evaluate(expression))
    assert np.isclose(
        evaluate(compile_form(sum(reads, num(0)), 4), env),
        sum(evaluate(compile_form(e, 4), env) for e in reads),
    )


def test_lowering_keeps_the_mesh_and_the_functions_of_the_user():
    # A lowering pass does not enter domains, domain maps or spaces, so the cells of the lowered
    # form are cells of the mesh of the user, and its function is the function of the user.
    D, _ = p1_mesh()
    uT = function(basix_space("V", T, P, 2), "u")
    out = lower_spatial(expand_integrals(functionals(D, push_forward(uT, D))[3]))
    # The function of the user is read through the cell maps of the topology of the user.
    reads = [n for n in nodes(out) if isinstance(n, PullBack) and n.expression == uT]
    assert reads and all(n.map.parent == T for n in reads)


def test_the_geometry_of_a_p1_coordinate_function_is_one_value_for_each_cell():
    # The derivative table of P1 does not vary over the points, so det J has no index of them.
    D, _ = p1_mesh()
    u = push_forward(function(basix_space("V", T, P, 2), "u"), D)
    compiled = compile_form(functionals(D, u)[3], 4)
    dets = {n.denominator for n in nodes(compiled) if isinstance(n, Division)}
    assert dets and all(d.free_indices == () for d in dets)


def test_curved_geometry_changes_with_only_the_edge_dofs():
    topology = grouped_topology((K,), (((0, 1, 2),),), ())
    X = basix_space("geometry", topology, P, 2, shape=(2,))
    x = function(X, "x")
    domain = mesh(x, EuclideanSpace(2))
    area = integral(num(1), domain)
    compiled = compile_form(area, 4)
    r, s = X.element(0).points.T
    # x(r, s) = (r, s + a*r*s) fixes all vertices and has determinant 1 + a*r.
    for a in (0.0, 1.0):
        dofs = np.column_stack((r, s + a * r * s)).ravel()
        expected = 0.5 + a / 6
        assert np.isclose(Interpreter({x: dofs}).evaluate(area), expected)
        assert np.isclose(evaluate(compiled, {"coefficients:x": dofs}), expected)


def test_the_jump_of_a_function_on_a_topology_across_a_facet():
    # A discontinuous function on T, restricted to the diagonal from both triangles, read on the
    # reference interval of each cell of the side topology.
    I = basix_cell(basix.CellType.interval)
    S = grouped_topology((I,), (((0, 2),),), ((("+", T, 0), ("-", T, 0)),))
    d = function(basix_space("W", T, P, 1, discontinuous=True), "d")
    jump = pull_back(d, side_map(T, S, "+")) - pull_back(d, side_map(T, S, "-"))
    f = integral(pull_back(jump, cell_map(S, 0)), I)
    W = np.array([0.0, 1.0, 2.0, 5.0, -1.0, 3.0])
    value = evaluate(compile_form(f, 2), {"coefficients:d": W})
    assert np.isclose(value, Interpreter({d: W}).evaluate(f))
    # Vertices 0 and 2 of the diagonal are dofs 0 and 2 of cell 0 and dofs 3 and 4 of cell 1.
    assert np.isclose(value, ((0.0 - 5.0) + (2.0 - -1.0)) / 2)
