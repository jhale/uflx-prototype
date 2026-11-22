"""Explicit side relations, independent orientation, and nested restrictions."""

import numpy as np
import pytest

from uflx.arithmetic import num
from uflx.calculus import coordinate, integral
from uflx.forms import function
from uflx.geometry import cell_map, conormal, domain_map, mesh, pull_back, side_map
from uflx.nodes import EuclideanSpace, ScalarType
from uflx.passes import expand_pull_backs
from uflx.tensors import tensor
from uflx_basix.nodes import TabulatedTopology
from uflx_basix.spaces import space
from uflx_compiler import compile_form, evaluate

from ..uflx_basix.test_side_relations import configuration


def test_orientation_changes_do_not_change_restriction_identity():
    _, _, _, _, D, E, A0, _, geometry = configuration()
    plus = side_map(E, (D[0], A0))
    aliases = {"+": plus, "air": plus}
    u = function(space("field", domain=D), "u")
    m = tensor((num(1), num(1)))
    outward, inward = conormal(plus, m), conormal(plus, -m)
    assert outward.side_map is inward.side_map is plus
    assert outward is not inward
    assert pull_back(u, aliases["+"]) is pull_back(u, aliases["air"])
    assert side_map(E, (D[0], A0)) is plus
    for component in range(2):
        a = evaluate(compile_form(integral(outward[component], E), 2), geometry)
        b = evaluate(compile_form(integral(inward[component], E), 2), geometry)
        np.testing.assert_allclose((a, b), (1.0, -1.0), atol=1e-12)


def test_source_geometry_is_independent_of_the_evaluation_relation():
    K, F, T, S, D, _, A0, _, geometry = configuration()
    # Deliberately translate the source geometry: field evaluation still follows A0.
    y = function(space("source-coordinates", (2,), S, ScalarType.real), "y")
    E = mesh(y, EuclideanSpace(2))
    plus = side_map(E, (D[0], A0))
    trace = pull_back(coordinate(D), plus)
    local = expand_pull_backs(pull_back(trace, cell_map(E, 0)))
    point = domain_map(EuclideanSpace(0), tensor((num(0.25),)), F)
    result = evaluate(compile_form(pull_back(local, point), 1), geometry)
    np.testing.assert_allclose(result, (0.75, 0.25))
    assert plus.entries[0][1].ambient_domain is K and plus.parent_topology is T


def test_nested_restrictions_select_repeated_slots_and_reverse_inclusions():
    _, F, _, _, D, _, A0, A1, geometry = configuration()
    S = TabulatedTopology("two-edges", (F, F))
    P = TabulatedTopology("reordered-edges", (F, F, F))
    identity = domain_map(F, coordinate(F), F)
    reverse = domain_map(F, tensor((num(1) - coordinate(F)[0],)), F)
    outer = side_map(S, (D[1], A1), (D[0], A0))
    inner = side_map(P, (S[1], reverse), (S[0], identity), (S[1], identity))
    on_edges = pull_back(coordinate(D), outer)
    reordered = pull_back(on_edges, inner)
    assert on_edges.domain is S and reordered.domain is P
    point = domain_map(EuclideanSpace(0), tensor((num(0.3),)), F)
    # Reversed first edge, second edge, and a repeated first edge.
    expected = ((0.3, 0.7), (0.7, 0.3), (0.7, 0.3))
    for i, coordinates in enumerate(expected):
        local = pull_back(reordered, cell_map(P, i))
        value = evaluate(compile_form(pull_back(local, point), 1), geometry)
        np.testing.assert_allclose(value, coordinates, atol=1e-12)
    other = TabulatedTopology("other-two-edges", (F, F))
    wrong_parent = side_map(P, (other[1], reverse), (other[0], identity), (other[1], identity))
    with pytest.raises(ValueError, match="pull-back target differs from expression domain"):
        pull_back(on_edges, wrong_parent)
