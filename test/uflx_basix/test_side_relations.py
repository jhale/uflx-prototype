"""Explicit side relations, independent orientation, and nested restrictions."""

import basix
import pytest

from uflx._src.algorithms.substitution import substitute
from uflx._src.nodes import AbstractTopology, SideMap
from uflx.arithmetic import num
from uflx.calculus import coordinate
from uflx.geometry import cell_map, conormal, domain_map, mesh, side_domain, side_map
from uflx.tensors import index, tensor
from uflx_basix.basix_nodes import BasixCell
from uflx_basix.nodes import StaticTopology, TabulatedTopology

from .meshes import vertex_mesh


def configuration():
    K = BasixCell(basix.CellType.triangle.value)
    F = BasixCell(basix.CellType.interval.value)
    T = StaticTopology((K, K), ((0, 1, 2), (1, 3, 2)))
    S = TabulatedTopology("edge", (F,))
    t = coordinate(F)[0]
    A0 = domain_map(F, tensor((num(1) - t, t)), K)
    A1 = domain_map(F, tensor((num(0), t)), K)
    D, geometry = vertex_mesh(T, ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)))
    E = side_domain(D, S, (0, A0))
    return K, F, T, S, D, E, A0, A1, geometry


def test_topology_requires_only_reference_cells():
    class Cells(AbstractTopology):
        identifier = "test.only_cells"
        cell_types = (BasixCell(basix.CellType.triangle.value),)

    T = Cells()
    assert T.dimension == 2
    assert not hasattr(T, "sides") and not hasattr(T, "side_data")
    assert cell_map(T, 0).reference_cell == T.cell_types[0]


def test_entries_and_conormals_validate_their_reference_domains():
    K, F, T, S, D, E, A0, _, _ = configuration()
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
    K, F, _T, S, D, _, A0, _, _ = configuration()
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
    _, F, T, _, D, E, A0, A1, _ = configuration()
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
