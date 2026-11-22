import basix
import numpy as np
import pytest

from uflx.arithmetic import num
from uflx.calculus import coordinate, integral
from uflx_basix.basix_cells import basix_cell
from uflx_basix.quadrature import apply_quadrature


def test_selected_degree_preserves_the_tensor_product_rule():
    cell = basix_cell(basix.CellType.quadrilateral)
    x = coordinate(cell)
    expr = integral(x[0] * x[1], cell)
    selected = apply_quadrature(expr, select=lambda _: np.int64(2))
    assert selected is apply_quadrature(expr, degree=2)


def test_rule_arguments_and_shapes():
    expr = integral(num(1), basix_cell(basix.CellType.triangle))
    with pytest.raises(TypeError, match="either"):
        apply_quadrature(expr, 2, select=lambda _: 3)
    with pytest.raises(ValueError, match="shape"):
        apply_quadrature(expr, select=lambda _: ([[0.5]], [1.0]))
