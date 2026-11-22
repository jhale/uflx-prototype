"""Numerical evaluation of gathered indices and external operators."""

import numpy as np
import pytest

from uflx import nodes
from uflx._src.extensions.external_operators import external_operator
from uflx.arithmetic import num, power
from uflx.calculus import coordinate
from uflx.nodes import AbstractExpression, EuclideanSpace
from uflx_basix.nodes import IndexTable
from uflx_oracle import evaluate

from ..external_operators import Energy


def test_a_gathered_index_reads_an_index_value_and_gathers_nest():
    i, j = nodes.Index("i"), nodes.Index("j")
    table = IndexTable(((0, 1), (1, 0)), 2)
    assert table.shape == (2, 2) and table.extent == 2
    assert not isinstance(table, AbstractExpression)
    with pytest.raises(TypeError):
        nodes.AbstractIndexTable()
    inner = nodes.GatheredIndex(IndexTable((1, 0), 2), (j,))
    g = nodes.GatheredIndex(table, (i, inner))
    assert g.extent == 2 and g.free_indices == ((i, 2), (j, 2))
    a = nodes.Indexed(nodes.Tensor((nodes.Float(10.0), nodes.Float(20.0))), (g,))
    assert a.free_indices == ((i, 2), (j, 2))
    # With i = 0 and j = 0, the inner table gives 1, the outer table gives table[0][1] = 1.
    assert evaluate(a, {}, {i: 0, j: 0})(()) == 20.0
    assert evaluate(a, {}, {i: 1, j: 0})(()) == 10.0


def test_oracle_can_supply_a_whole_field_to_an_external_operator():
    P = EuclideanSpace(1)
    x = coordinate(P)[0]
    f = external_operator(Energy(), (x, power(num(2), num(-1))))
    env = {Energy(): lambda field, scale: field(np.array([0.25])) * scale}
    assert evaluate(f, env)(()) == 0.125
