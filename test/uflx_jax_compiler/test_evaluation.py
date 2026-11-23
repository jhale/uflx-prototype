"""Numerical evaluation of gathered indices and external operators."""

import numpy as np
import pytest

from uflx import nodes
from uflx.arithmetic import num, power
from uflx.calculus import coordinate
from uflx.nodes import AbstractExpression, EuclideanSpace
from uflx_extensions.external_operators import external_operator
from uflx_extensions.index_tables import index_table
from uflx_jax_compiler import trace

from ..uflx.library import IndexTable
from ..uflx_extensions.operations import Energy


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
    assert trace(a, i, j)(0, 0) == 20.0
    assert trace(a, i, j)(1, 0) == 10.0


def test_scalar_and_rectangular_runtime_index_tables():
    cell = index_table("cell", shape=(), extent=2)
    dofmap = index_table("dofmap", shape=(2, 3), extent=4)
    values = nodes.Tensor(tuple(num(value) for value in (10, 20, 30, 40)))
    read = trace(values[dofmap[cell[()], 2]], cell, dofmap)
    cell_dofs = np.array([[0, 1, 2], [2, 1, 3]])
    assert read(0, cell_dofs) == 30
    assert read(1, cell_dofs) == 40


def test_oracle_can_supply_a_whole_field_to_an_external_operator():
    P = EuclideanSpace(1)
    x = coordinate(P)[0]
    f = external_operator(Energy(), (x, power(num(2), num(-1))))
    env = {Energy(): lambda field, scale: field(np.array([0.25])) * scale}
    assert trace(f, *env)(*env.values()) == 0.125


def test_external_operator_can_sample_its_field_after_the_call():
    from uflx.forms import apply, form, function

    from ..uflx.spaces import space
    from ..uflx_basix.integrals import D

    V = space("cell field", domain=D, scalar_type="real")
    u, v = function(V, "u"), function(V, "v")
    fields = []

    def remember(field, scale):
        fields.append(field)
        return scale

    energy = external_operator(Energy(), (v, num(1)))
    expression = apply(form(energy, (v,)), (u,))
    kernel = trace(expression, Energy(), u)
    for coefficient in (lambda p: p[0], lambda p: 2 + p[0]):
        assert kernel(remember, coefficient) == 1

    # Each saved operand retains the coefficient selected by the caller.
    assert [field(np.array([0.25, 0.25])) for field in fields] == [0.25, 2.25]
