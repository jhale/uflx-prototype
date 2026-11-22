import basix
import numpy as np
import pytest

from uflx.arithmetic import num
from uflx.calculus import coordinate, integral
from uflx.tensors import index, tensor
from uflx_basix.cells import cell
from uflx_basix.quadrature import apply_quadrature
from uflx_oracle.interpreter import Interpreter


def test_choose_rules_for_distinct_integrals_on_the_same_cell():
    K = cell(basix.CellType.interval)
    x = coordinate(K)[0]
    a, b, c = (integral(x ** num(power), K) for power in (2, 3, 4))
    expression = tensor((a, b, c, a))
    rules = {a: ([[0.25]], [1.0]), b: 4}
    visited = []

    def choose(node):
        visited.append(node)
        return rules.get(node)

    partial = apply_quadrature(expression, select=choose)
    assert visited == [a, b, c]
    assert partial.entries[2] is c
    assert partial.entries[0] is partial.entries[3]
    np.testing.assert_allclose(Interpreter().evaluate(partial), [1 / 16, 1 / 4, 1 / 5, 1 / 16])

    finished = apply_quadrature(partial, select={c: 1}.get)
    assert finished.entries[:2] == partial.entries[:2]
    np.testing.assert_allclose(Interpreter().evaluate(finished), [1 / 16, 1 / 4, 1 / 16, 1 / 16])


def test_nested_rules_are_selected_by_original_integral():
    K = cell(basix.CellType.interval)
    x = coordinate(K)[0]
    inner = integral(x * x, K)
    outer = integral(inner * x, K)
    rules = {inner: ([[1.0]], [1.0]), outer: ([[0.25]], [1.0])}
    result = apply_quadrature(outer, select=rules.get)
    assert Interpreter().evaluate(result) == 0.25
    assert apply_quadrature(outer, select=lambda _: None) is outer


def test_custom_rule_preserves_tensor_and_free_axes():
    K = cell(basix.CellType.triangle)
    x = coordinate(K)
    i = index("row")
    expr = integral(tensor((num(1), num(2)))[i] * x, K)
    points, weights = basix.make_quadrature(K.cell_type, 2)
    quadrature = apply_quadrature(expr, select=lambda _: (points, weights))
    assert (
        quadrature.value_shape == expr.value_shape and quadrature.free_indices == expr.free_indices
    )
    np.testing.assert_allclose(Interpreter().evaluate(quadrature), [[1 / 6, 1 / 6], [1 / 3, 1 / 3]])
    empty = apply_quadrature(expr, select=lambda _: (points[:0], weights[:0]))
    assert empty.free_indices == expr.free_indices
    np.testing.assert_array_equal(Interpreter().evaluate(empty), np.zeros((2, 2)))


def test_explicit_weights_on_a_point_cell():
    expr = integral(num(3), cell(basix.CellType.point))
    result = apply_quadrature(expr, select=lambda _: (np.empty((2, 0)), [0.5, 1.5]))
    assert Interpreter().evaluate(result) == 6


@pytest.mark.parametrize("cell_type", [basix.CellType.quadrilateral, basix.CellType.hexahedron])
def test_basix_rule_on_a_tensor_product_cell(cell_type):
    K = cell(cell_type)
    x = coordinate(K)
    moment = integral(x[0] * x[0] + x[K.dimension - 1], K)
    assert Interpreter().evaluate(apply_quadrature(moment, degree=2)) == pytest.approx(5 / 6)
