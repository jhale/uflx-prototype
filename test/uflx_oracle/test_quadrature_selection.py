import basix
import numpy as np
import pytest

from uflx.algorithms import bottom_up, substitute_values
from uflx.arithmetic import num
from uflx.calculus import coordinate, integral
from uflx.nodes import Integral, PullBack
from uflx.passes import expand_pull_backs
from uflx.tensors import index, tensor
from uflx_basix.cells import cell
from uflx_basix.quadrature import apply_quadrature
from uflx_oracle.interpreter import Interpreter

from ..testutil import contains


def test_choose_rules_for_distinct_integrals_on_the_same_cell():
    K = cell(basix.CellType.interval)
    x = coordinate(K)[0]
    a, b, c = (integral(x ** num(power), K) for power in (2, 3, 4))
    expression = tensor((a, b, c, a))
    partial = substitute_values(
        expression,
        {
            a: apply_quadrature(a, [[0.25]], [1.0]),
            b: apply_quadrature(b, *K.quadrature(4)),
        },
    )
    assert partial.entries[2] is c
    assert partial.entries[0] is partial.entries[3]
    np.testing.assert_allclose(Interpreter().evaluate(partial), [1 / 16, 1 / 4, 1 / 5, 1 / 16])

    finished = substitute_values(partial, {c: apply_quadrature(c, *K.quadrature(1))})
    assert finished.entries[:2] == partial.entries[:2]
    np.testing.assert_allclose(Interpreter().evaluate(finished), [1 / 16, 1 / 4, 1 / 16, 1 / 16])


def test_one_integral_or_a_bottom_up_traversal_of_nested_integrals():
    K = cell(basix.CellType.interval)
    x = coordinate(K)[0]
    inner = integral(x * x, K)
    outer = integral(inner * x, K)
    rule = ([[0.25]], [1.0])
    # Sampling just the outer integral leaves the inner integral equal to 1/3.
    assert Interpreter().evaluate(apply_quadrature(outer, *rule)) == pytest.approx(1 / 12)

    def sample(node):
        if isinstance(node, Integral) and node.integration_domain == K:
            return apply_quadrature(node, *rule)
        return node

    assert Interpreter().evaluate(bottom_up(outer, sample)) == pytest.approx(1 / 64)


def test_point_evaluations_can_be_expanded_after_sampling():
    K = cell(basix.CellType.triangle)
    moment = integral(coordinate(K)[0], K)
    sampled = apply_quadrature(moment, *K.quadrature(2))
    assert contains(sampled, lambda node: isinstance(node, PullBack))
    expanded = expand_pull_backs(sampled)
    assert not contains(expanded, lambda node: isinstance(node, PullBack))
    assert Interpreter().evaluate(expanded) == pytest.approx(1 / 6)


def test_custom_rule_preserves_tensor_and_free_axes():
    K = cell(basix.CellType.triangle)
    x = coordinate(K)
    i = index("row")
    expr = integral(tensor((num(1), num(2)))[i] * x, K)
    points, weights = basix.make_quadrature(K.cell_type, 2)
    quadrature = apply_quadrature(expr, points, weights)
    assert (
        quadrature.value_shape == expr.value_shape and quadrature.free_indices == expr.free_indices
    )
    np.testing.assert_allclose(Interpreter().evaluate(quadrature), [[1 / 6, 1 / 6], [1 / 3, 1 / 3]])
    empty = apply_quadrature(expr, points[:0], weights[:0])
    assert empty.free_indices == expr.free_indices
    np.testing.assert_array_equal(Interpreter().evaluate(empty), np.zeros((2, 2)))


def test_explicit_weights_on_a_point_cell():
    expr = integral(num(3), cell(basix.CellType.point))
    result = apply_quadrature(expr, np.empty((2, 0)), [0.5, 1.5])
    assert Interpreter().evaluate(result) == 6


@pytest.mark.parametrize("cell_type", [basix.CellType.quadrilateral, basix.CellType.hexahedron])
def test_basix_rule_on_a_tensor_product_cell(cell_type):
    K = cell(cell_type)
    x = coordinate(K)
    moment = integral(x[0] * x[0] + x[K.dimension - 1], K)
    assert Interpreter().evaluate(apply_quadrature(moment, *K.quadrature(2))) == pytest.approx(
        5 / 6
    )
