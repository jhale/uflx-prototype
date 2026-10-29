import basix
import numpy as np
from test_basix import CELLS, square
from test_compiler import functionals, runtime_mesh

from uflx_compiler import compile_form, evaluate
from uflx_oracle import evaluate as oracle
from uflx_spec_interpreter import (
    RD,
    apply_gradients,
    cell_index,
    expand_integrals,
    function,
    gathered_index,
    index,
    index_sum,
    integral,
    pull_back,
    tensor,
)
from uflx_spec_interpreter._draft import cell_domain, cell_map, coordinate_function, mesh
from uflx_spec_interpreter._src.domains import is_cell
from uflx_spec_interpreter._src.nodes import CellMap, Division, PullBack
from uflx_spec_interpreter._src.passes import gradients
from uflx_spec_interpreter._src.substitution import everywhere
from uflx_toy.basix_cells import basix_cell
from uflx_toy.basix_spaces import basix_space
from uflx_toy.indices import index_table
from uflx_toy.topology import static_topology

P = basix.ElementFamily.P
K = basix_cell(CELLS[2])
# The unit square as two triangles, with the vertices of square(K) of test_basix.
T = static_topology((K,), (((0, 1, 2), (0, 2, 3)),), ())
VERTICES = np.array([(0, 0), (1, 0), (1, 1), (0, 1)], dtype=float)


def coordinates(degree, discontinuous=False):
    X = basix_space("X", T, P, degree, discontinuous)
    x0, x1 = function(X, "x0"), function(X, "x1")
    return X, x0, x1, tensor((x0, x1))


def p1_mesh():
    # The dofs of a P1 space on T are numbered by the vertices, in the order of their first cell.
    X, x0, x1, x = coordinates(1)
    assert X.dofmap == (((0, 1, 2), (0, 2, 3)),)
    return mesh(x, RD(2)), {x0: VERTICES[:, 0], x1: VERTICES[:, 1]}


def labelled(env):
    return {f"coefficients:{f.label}": v for f, v in env.items()}


def nodes(e):
    found = []
    everywhere(e, lambda n: found.append(n) or True)
    return found


def test_a_mesh_from_a_coordinate_function_has_the_topology_and_the_cells_of_the_draft():
    D, _ = p1_mesh()
    x = coordinate_function(D)
    assert D.topology == T and x.support == T
    g = gathered_index(index_table((1, 0), 2), (index("i"),))
    C = cell_domain(D, 0, g)
    assert C.parameterisation == pull_back(x, CellMap(T, 0, g)) and is_cell(C)
    assert cell_map(T, 0) == CellMap(T, 0, cell_index(T, 0))


def test_a_p1_coordinate_function_gives_the_values_of_the_explicit_mesh():
    D, geometry = p1_mesh()
    E = square(K)[0]
    U = np.random.default_rng(0).normal(size=basix_space("V", D, P, 2).num_dofs)
    u, w = function(basix_space("V", D, P, 2), "u"), function(basix_space("V", E, P, 2), "u")
    for f, g in zip(functionals(D, u), functionals(E, w), strict=True):
        compiled = compile_form(f, 4)
        expected = evaluate(compile_form(g, 4), {"coefficients:u": U})
        assert np.isclose(evaluate(compiled, labelled(geometry) | {"coefficients:u": U}), expected)
        assert np.isclose(oracle(f, geometry | {u: U})(()), expected)


def test_the_jacobian_of_a_mesh_is_the_reference_gradient_of_the_coordinate_function():
    D, _ = p1_mesh()
    u = function(basix_space("V", D, P, 2), "u")
    out = gradients.lower(expand_integrals(functionals(D, u)[3]))
    assert gradients.in_n6(out)
    reads = [n for n in nodes(out) if isinstance(n, PullBack) and isinstance(n.domain, CellMap)]
    assert {n.expression.label for n in reads} == {"x0", "x1"}


def test_a_curved_p2_coordinate_function_agrees_with_the_oracle():
    X, x0, x1, x = coordinates(2)
    D = mesh(x, RD(2))
    # Interpolate the square, then bend the shared diagonal.
    points = basix.create_element(P, CELLS[2], 2, basix.LagrangeVariant.equispaced).points
    values = np.zeros((X.num_dofs, 2))
    for c, rows in enumerate(X.dofmap[0]):
        a, b, d = VERTICES[list(T.cells[0][c])]
        values[list(rows)] = a + np.outer(points[:, 0], b - a) + np.outer(points[:, 1], d - a)
    midpoint = next(d for d in X.dofmap[0][0] if np.allclose(values[d], (0.5, 0.5)))
    values[midpoint] += (0.08, -0.05)
    geometry = {x0: values[:, 0], x1: values[:, 1]}
    u = function(basix_space("V", D, P, 2), "u")
    U = np.random.default_rng(1).normal(size=u.space.num_dofs)
    # The integrands are rational on a curved cell, so the rule has a high degree.
    for f in functionals(D, u):
        compiled = compile_form(f, 16)
        value = evaluate(compiled, labelled(geometry) | {"coefficients:u": U})
        assert np.isclose(value, oracle(f, geometry | {u: U})(()), rtol=1e-9), f


def test_a_discontinuous_coordinate_function_from_the_vertices_of_each_cell():
    # The coordinates of the vertices of each cell are the dofs of a discontinuous P1 space.
    X, _, _, x = coordinates(1, discontinuous=True)
    D = mesh(x, RD(2))
    a, _, E = runtime_mesh()
    rng = np.random.default_rng(12)
    corners = np.array([(0, 0), (1, 0), (1.2, 1.1), (-0.1, 0.9)]) + rng.normal(size=(4, 2)) * 0.05
    cells = corners[[[0, 1, 2], [0, 2, 3]]]
    flat = cells.reshape(-1, 2)[np.argsort(np.ravel(X.dofmap[0]))]
    geometry = {"coefficients:x0": flat[:, 0], "coefficients:x1": flat[:, 1]}
    U = rng.normal(size=basix_space("V", D, P, 2).num_dofs)
    u, w = function(basix_space("V", D, P, 2), "u"), function(basix_space("V", E, P, 2), "u")
    for f, g in zip(functionals(D, u), functionals(E, w), strict=True):
        value = evaluate(compile_form(f, 4), geometry | {"coefficients:u": U})
        expected = evaluate(compile_form(g, 4), {a.label: cells, "coefficients:u": U})
        assert np.isclose(value, expected)


def test_a_gathered_cell_of_a_mesh_reads_the_cells_that_the_entry_selects():
    D, geometry = p1_mesh()
    u = function(basix_space("V", D, P, 2), "u")
    U = np.random.default_rng(2).normal(size=u.space.num_dofs)
    i, c = index("i"), cell_index(T, 0)
    g = gathered_index(index_table((1, 0), 2), (i,))
    gathered = index_sum(integral(pull_back(u, cell_domain(D, 0, g)), K), (i,))
    plain = index_sum(integral(pull_back(u, cell_domain(D, 0)), K), (c,))
    env = labelled(geometry) | {"coefficients:u": U}
    value = evaluate(compile_form(gathered, 4), env)
    assert np.isclose(value, evaluate(compile_form(plain, 4), env))
    assert np.isclose(value, oracle(gathered, geometry | {u: U})(()))
    assert gradients.in_n6(apply_gradients(gathered))


def test_lowering_keeps_the_mesh_and_the_functions_of_the_user():
    # A lowering pass does not enter domains, domain maps or spaces, so the cells of the lowered
    # form are cells of the mesh of the user, and its function is the function of the user.
    D, _ = p1_mesh()
    u = function(basix_space("V", D, P, 2), "u")
    out = gradients.lower(expand_integrals(functionals(D, u)[3]))
    reads = [n for n in nodes(out) if isinstance(n, PullBack) and n.expression == u]
    assert reads and all(n.domain.ambient_domain == D for n in reads)


def test_the_geometry_of_a_p1_coordinate_function_is_one_value_for_each_cell():
    # The derivative table of P1 does not vary over the points, so det J has no index of them.
    D, _ = p1_mesh()
    u = function(basix_space("V", D, P, 2), "u")
    compiled = compile_form(functionals(D, u)[3], 4)
    dets = {n.second for n in nodes(compiled) if isinstance(n, Division)}
    assert dets and all([i for i, _ in d.free_indices] == [cell_index(T, 0)] for d in dets)
