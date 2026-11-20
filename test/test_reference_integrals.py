"""Selected entity integrals share mesh expansion without introducing a binder."""

import numpy as np
import pytest
from test_facets import E_bnd, E_int, sides_int
from test_integrals import D, K
from testutil import grouped_topology, vertex_mesh

from uflx_oracle import evaluate
from uflx_spec_interpreter import (
    coordinate,
    expand_integrals,
    index,
    indexed,
    integral,
    lower,
    num,
    reference_integral,
    tensor,
)
from uflx_spec_interpreter._src.algorithms.substitution import substitute
from uflx_spec_interpreter._src.nodes import IndexSum, Integral
from uflx_toy.sides import outward_conormal


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
        evaluate(lower(reference_integral(f, domain, entry)), {})(())
        for entry in range(len(domain.topology.cell_types))
    ]
    assert np.allclose(sum(pieces), whole)


def test_selected_group_and_tensor_value():
    topology = grouped_topology((K, K), (((0, 1, 2),), ((0, 2, 3),)), ())
    mesh = vertex_mesh(topology, [(0, 0), (2, 0), (2, 1), (0, 3)])
    f = tensor((num(1), num(2)))
    for group, area in enumerate((1, 3)):
        piece = reference_integral(f, mesh, group)
        assert isinstance(piece, Integral)
        assert piece.value_shape == (2,)
        assert np.allclose(evaluate(lower(piece), {})(()), (area, 2 * area))


def test_expansion_preserves_array_indices_without_a_cell_binder():
    c = index("load-entry")
    f = indexed(tensor((num(2), num(3))), (c,))
    piece = reference_integral(f, D, 0)
    assert dict(piece.free_indices) == {c: 2}
    expanded = expand_integrals(integral(f, D))
    assert not isinstance(expanded, IndexSum)
    assert dict(expanded.free_indices) == {c: 2}
    for entry in (0, 1):
        expected = evaluate(lower(integral(substitute(f, {c: entry}), D)), {})(())
        assert np.isclose(evaluate(lower(expanded), {}, {c: entry})(()), expected)


def test_interior_facet_keeps_both_cell_traces():
    jump = outward_conormal(sides_int["+"])[0] - outward_conormal(sides_int["-"])[0]
    piece = reference_integral(jump * jump, E_int, 0)
    assert np.allclose(evaluate(lower(piece), {})(()), 2 * np.sqrt(2))


def test_fixed_slot_selects_the_requested_cell():
    mesh = vertex_mesh(D.topology, [(0, 0), (2, 0), (2, 1), (0, 3)])
    for slot, area in ((1, 3), (0, 1)):
        piece = reference_integral(num(1), mesh, slot)
        assert piece.free_indices == ()
        assert np.allclose(evaluate(lower(piece), {})(()), area)
