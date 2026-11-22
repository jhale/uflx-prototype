"""Explicit side relations, independent orientation, and nested restrictions."""

import basix
import numpy as np
import pytest

from uflx._src.algorithms.substitution import substitute
from uflx._src.nodes import AbstractTopology, SideMap
from uflx.arithmetic import num
from uflx.calculus import coordinate, integral
from uflx.forms import function
from uflx.geometry import cell_map, conormal, domain_map, mesh, pull_back, side_domain, side_map
from uflx.nodes import EuclideanSpace, ScalarType
from uflx.passes import expand_pull_backs
from uflx.tensors import index, tensor
from uflx_basix.basix_nodes import BasixCell
from uflx_basix.nodes import StaticTopology, TabulatedTopology, VertexCoordinates
from uflx_basix.spaces import space
from uflx_compiler import compile_form, evaluate


def configuration():
    K = BasixCell(basix.CellType.triangle.value)
    F = BasixCell(basix.CellType.interval.value)
    T = StaticTopology((K, K), ((0, 1, 2), (1, 3, 2)))
    S = TabulatedTopology("edge", (F,))
    t = coordinate(F)[0]
    A0 = domain_map(F, tensor((num(1) - t, t)), K)
    A1 = domain_map(F, tensor((num(0), t)), K)
    x = VertexCoordinates(T, ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)))
    D = mesh(x, EuclideanSpace(2))
    E = side_domain(D, S, (0, A0))
    return K, F, T, S, D, E, A0, A1


def test_topology_requires_only_reference_cells():
    class Cells(AbstractTopology):
        identifier = "test.only_cells"
        cell_types = (BasixCell(basix.CellType.triangle.value),)

    T = Cells()
    assert T.dimension == 2
    assert not hasattr(T, "sides") and not hasattr(T, "side_data")
    assert cell_map(T, 0).reference_cell == T.cell_types[0]


def test_orientation_changes_do_not_change_restriction_identity():
    _, _, _, _, D, E, A0, _ = configuration()
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
        a = evaluate(compile_form(integral(outward[component], E), 2), {})
        b = evaluate(compile_form(integral(inward[component], E), 2), {})
        np.testing.assert_allclose((a, b), (1.0, -1.0), atol=1e-12)


def test_source_geometry_is_independent_of_the_evaluation_relation():
    K, F, T, S, D, _, A0, _ = configuration()
    # Deliberately translate the source geometry: field evaluation still follows A0.
    y = function(space("source-coordinates", (2,), S, ScalarType.real), "y")
    E = mesh(y, EuclideanSpace(2))
    plus = side_map(E, (D[0], A0))
    trace = pull_back(coordinate(D), plus)
    local = expand_pull_backs(pull_back(trace, cell_map(E, 0)))
    point = domain_map(EuclideanSpace(0), tensor((num(0.25),)), F)
    result = evaluate(compile_form(pull_back(local, point), 1), {})
    np.testing.assert_allclose(result, (0.75, 0.25))
    assert plus.entries[0][1].ambient_domain is K and plus.parent_topology is T


def test_nested_restrictions_select_repeated_slots_and_reverse_inclusions():
    _, F, _, _, D, _, A0, A1 = configuration()
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
        value = evaluate(compile_form(pull_back(local, point), 1), {})
        np.testing.assert_allclose(value, coordinates, atol=1e-12)
    other = TabulatedTopology("other-two-edges", (F, F))
    wrong_parent = side_map(P, (other[1], reverse), (other[0], identity), (other[1], identity))
    with pytest.raises(ValueError, match="pull-back target differs from expression domain"):
        pull_back(on_edges, wrong_parent)


def test_entries_and_conormals_validate_their_reference_domains():
    K, F, T, S, D, E, A0, _ = configuration()
    with pytest.raises(ValueError, match="per source slot"):
        side_map(E, (D[0], A0), (D[1], A0))
    with pytest.raises(ValueError, match="inclusion target"):
        side_map(S, (T[0], domain_map(F, coordinate(F), F)))
    relation = side_map(E, (D[0], A0))
    m = tensor((num(1), num(1)))
    with pytest.raises(ValueError, match="per source slot"):
        conormal(relation)
    with pytest.raises(ValueError, match="wrong shape"):
        conormal(relation, num(1))
    with pytest.raises(ValueError, match="wrong domain"):
        conormal(relation, coordinate(K))
    with pytest.raises(ValueError, match="real field"):
        conormal(relation, num(1j) * m)
    with pytest.raises(ValueError, match="mesh side"):
        conormal(side_map(S, (T[0], A0)), m)


def test_reconstruction_and_fixed_relation_data():
    K, F, _T, S, D, _, A0, _ = configuration()
    i = index("relation-data")
    factors = tensor((num(1), num(2)))[i]
    A = domain_map(F, factors * A0.parameterisation, K)
    with pytest.raises(ValueError, match="no free indices"):
        side_map(S, (D[0], A))
    relation = side_map(S, (D[0], A0))
    m = tensor((num(1), num(1)))
    with pytest.raises(ValueError, match="no free indices"):
        conormal(relation, factors * m)
    n = conormal(relation, m)
    assert relation.free_indices == n.free_indices == ()
    A1 = domain_map(F, tensor((num(0), coordinate(F)[0])), K)
    assert substitute(relation, {A0: A1}) is side_map(S, (D[0], A1))
    assert substitute(n, {m: -m}).side_map is relation


def test_side_factory_uses_cell_handles_and_requires_a_common_parent():
    _, F, T, _, D, E, A0, A1 = configuration()
    left, right = D[0], D[1]
    source = TabulatedTopology("three-source-slots", (F, F, F))
    relation = side_map(source, (right, A1), (left, A0), (right, A1))
    assert relation is SideMap(D, source, ((1, A1), (0, A0), (1, A1)))
    other = mesh(num(2) * D.coordinate_function, D.ambient_domain)
    for wrong_parent in (T[0], other[0]):
        with pytest.raises(ValueError, match="common parent"):
            side_map(source, (left, A0), (wrong_parent, A0), (right, A1))
    with pytest.raises(TypeError, match="CellMap handles"):
        side_map(E, (0, A0))
    with pytest.raises(ValueError, match="at least one"):
        side_map(E)
