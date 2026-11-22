import basix
import numpy as np
import pytest

from uflx.arithmetic import num
from uflx.calculus import coordinate, integral
from uflx.geometry import domain_map, pull_back
from uflx.nodes import EuclideanSpace
from uflx.tensors import tensor
from uflx_basix.cells import cell
from uflx_basix.quadrature import apply_quadrature


def test_selecting_a_degree_uses_the_same_rule_as_a_fixed_degree():
    K = cell(basix.CellType.quadrilateral)
    x = coordinate(K)
    expr = integral(x[0] * x[1], K)
    selected = apply_quadrature(expr, select=lambda _: np.int64(2))
    assert selected is apply_quadrature(expr, degree=2)


def test_rule_arguments_and_shapes():
    expr = integral(num(1), cell(basix.CellType.triangle))
    with pytest.raises(TypeError, match="either"):
        apply_quadrature(expr, 2, select=lambda _: 3)
    with pytest.raises(ValueError, match="shape"):
        apply_quadrature(expr, select=lambda _: ([[0.5]], [1.0]))


def test_fixed_degree_only_rewrites_reference_integrals():
    K = cell(basix.CellType.interval)
    point = domain_map(EuclideanSpace(0), tensor((num(0.5),)), K)
    value = pull_back(coordinate(K)[0], point)
    assert apply_quadrature(value, degree=2) is value
