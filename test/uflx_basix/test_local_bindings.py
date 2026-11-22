"""Local element coefficients and library continuity are independent choices."""

import basix
import numpy as np
import pytest

from uflx.arithmetic import num
from uflx.calculus import integral
from uflx.forms import form, function
from uflx_basix.assembly import DofMap, conforming_dofmap
from uflx_basix.basix_cells import basix_cell
from uflx_basix.basix_spaces import basix_space
from uflx_compiler import assemble, assemble_bound

from ..testutil import push_forward
from .test_basix import square


def test_shared_vertices_do_not_identify_local_coefficients():
    D = square(basix_cell(basix.CellType.triangle))[0]
    V = basix_space("V", D.topology, basix.ElementFamily.P, 1)
    u, v, q = (function(V, name) for name in ("u", "v", "q"))
    U, W, Q = (push_forward(f, D) for f in (u, v, q))
    f = form(integral(U * W, D), (v, u))
    local = assemble(f, 2)({})
    assert V.cell_dofs == ((0, 1, 2), (3, 4, 5))
    assert local.shape == (6, 6)
    np.testing.assert_array_equal(local[:3, 3:], 0)

    # Exactly the same expression supports conforming or independent global DOFs.
    shared = conforming_dofmap(V)
    separate = DofMap(tuple(range(6)), 6)
    assert shared.indices == (0, 1, 2, 0, 2, 3)
    for mapping in (shared, separate):
        assembled = assemble_bound(f, 2, {u: mapping, v: mapping})({})
        global_values = np.arange(mapping.num_dofs, dtype=float)
        packed = mapping.gather(global_values)
        np.testing.assert_allclose(
            global_values @ assembled @ global_values, packed @ local @ packed
        )
        load = form(integral(Q * W, D), (v,))
        actual = assemble_bound(load, 2, {q: mapping, v: mapping})(
            {"coefficients:q": global_values}
        )
        np.testing.assert_allclose(actual, assembled @ global_values)


def test_rectangular_zero_and_independent_argument_bindings():
    D = square(basix_cell(basix.CellType.triangle))[0]
    V = basix_space("V", D.topology, basix.ElementFamily.P, 1)
    u, v = function(V, "u"), function(V, "v")
    row_map = DofMap((0, 1, 2, 0, 2, 3), 4)
    column_map = DofMap((0, 1, 2, 3, 4, 5), 7)  # An unused library DOF is retained.
    zero = form(integral(num(0), D), (v, u))
    result = assemble_bound(zero, 2, {v: row_map, u: column_map})({})
    np.testing.assert_array_equal(result, np.zeros((4, 7)))
    U, W = push_forward(u, D), push_forward(v, D)
    mass = form(integral(U * W, D), (v, u))
    local = assemble(mass, 2)({})
    expected = row_map.scatter(column_map.scatter(local, axis=1), axis=0)
    np.testing.assert_allclose(assemble_bound(mass, 2, {v: row_map, u: column_map})({}), expected)


@pytest.mark.parametrize("shape", [(2,), (2, 2)])
def test_blocked_fields_keep_cell_and_component_order(shape):
    D = square(basix_cell(basix.CellType.triangle))[0]
    V = basix_space("V", D.topology, basix.ElementFamily.P, 1, shape=shape)
    mapping = conforming_dofmap(V)
    values = np.arange(mapping.num_dofs)
    packed = mapping.gather(values).reshape(2, 3, -1)
    np.testing.assert_array_equal(packed[0, 0], packed[1, 0])
    np.testing.assert_array_equal(packed[0, 2], packed[1, 1])
    assert V.num_dofs == 6 * np.prod(shape)


def test_one_library_field_can_bind_different_local_input_roles():
    from ..testutil import static_topology, vertex_mesh

    K = basix_cell(basix.CellType.triangle)
    T = static_topology((K,), ((0, 1, 2),), ())
    D = vertex_mesh(T, [(0, 0), (1, 0), (0, 1)])
    V = basix_space("temperature", T, basix.ElementFamily.P, 1)
    first, second = function(V, "first-role"), function(V, "second-role")
    a = integral(push_forward(first, D), D)
    b = integral(push_forward(second, D), D)
    first_map = DofMap((0, 1, 2), 4)
    second_map = DofMap((1, 3, 2), 4)
    library_field = np.array([2.0, 3.0, 5.0, 7.0])
    result_a = assemble_bound(a, 2, {first: first_map})({"coefficients:first-role": library_field})
    result_b = assemble_bound(b, 2, {second: second_map})(
        {"coefficients:second-role": library_field}
    )
    # A P1 integral on the unit triangle is the sum of its three values divided by 6.
    np.testing.assert_allclose((result_a, result_b), (10 / 6, 15 / 6))
