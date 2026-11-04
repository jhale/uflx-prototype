"""Selected entity integrals share mesh expansion without introducing a binder."""

import numpy as np
import pytest
from test_facets import E_bnd, E_int, sides_int
from test_integrals import D, K
from testutil import vertex_mesh

from uflx_oracle import evaluate
from uflx_spec_interpreter import (
    cell_index,
    coordinate,
    expand_integrals,
    gathered_index,
    index,
    indexed,
    integral,
    lower,
    normal,
    num,
    reference_integral,
    tensor,
)
from uflx_spec_interpreter._src.algorithms.substitution import substitute
from uflx_spec_interpreter._src.nodes import IndexSum, Integral
from uflx_toy.indices import index_table
from uflx_toy.topology import static_topology


@pytest.mark.parametrize("domain", [D, E_bnd, E_int])
@pytest.mark.parametrize("support", ["constant", "mesh", "ambient"])
def test_selected_entities_sum_to_the_mesh_integral(domain, support):
    f = (
        num(2)
        if support == "constant"
        else coordinate(domain if support == "mesh" else domain.ambient_domain)[0]
    )
    whole = evaluate(lower(integral(f, domain)), {})(())
    pieces = [
        evaluate(lower(reference_integral(f, domain, 0, entry)), {})(())
        for entry in range(domain.topology.num_cells[0])
    ]
    assert np.allclose(sum(pieces), whole)


def test_selected_group_and_tensor_value():
    topology = static_topology((K, K), (((0, 1, 2),), ((0, 2, 3),)), ())
    mesh = vertex_mesh(topology, [(0, 0), (2, 0), (2, 1), (0, 3)])
    f = tensor((num(1), num(2)))
    for group, area in enumerate((1, 3)):
        piece = reference_integral(f, mesh, group, 0)
        assert isinstance(piece, Integral)
        assert piece.value_shape == (2,)
        assert np.allclose(evaluate(lower(piece), {})(()), (area, 2 * area))


def test_expansion_freshens_but_selected_entries_remain_free():
    c = cell_index(D.topology, 0)
    f = indexed(tensor((num(2), num(3))), (c,))
    entry = index("selected_cell")
    piece = reference_integral(f, D, 0, entry)
    assert dict(piece.free_indices) == {c: 2, entry: 2}
    assert substitute(piece, {entry: 0}) is reference_integral(f, D, 0, 0)
    expanded = expand_integrals(integral(f, D))
    assert isinstance(expanded, IndexSum)
    assert expanded.indices != (c,)
    assert dict(expanded.free_indices) == {c: 2}
    assert expanded.expression is reference_integral(f, D, 0, expanded.indices[0])


def test_interior_facet_keeps_both_cell_traces():
    jump = normal(sides_int["+"])[0] - normal(sides_int["-"])[0]
    piece = reference_integral(jump * jump, E_int, 0, 0)
    assert np.allclose(evaluate(lower(piece), {})(()), 2 * np.sqrt(2))


def test_gathered_entry_selects_the_requested_cell():
    mesh = vertex_mesh(D.topology, [(0, 0), (2, 0), (2, 1), (0, 3)])
    i = index("selection")
    entry = gathered_index(index_table((1, 0), 2), (i,))
    piece = reference_integral(num(1), mesh, 0, entry)
    assert dict(piece.free_indices) == {i: 2}
    for selection, area in enumerate((3, 1)):
        assert np.allclose(evaluate(lower(piece), {}, {i: selection})(()), area)
