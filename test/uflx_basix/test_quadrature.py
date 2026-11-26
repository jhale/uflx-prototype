# Copyright (C) 2026 Jack S. Hale
#
# This file is part of uflx-core-experimental (https://fenics.github.io/uflx/language/)
#
# SPDX-License-Identifier:    MIT

import basix
import pytest

from uflx.arithmetic import num
from uflx.calculus import coordinate, integral
from uflx.forms import form
from uflx.nodes import EuclideanSpace
from uflx.tensors import index
from uflx_basix.cells import cell
from uflx_basix.evaluation import evaluate_at_points
from uflx_basix.quadrature import apply_quadrature, weighted_sum


def test_quadrature_requires_one_reference_integral():
    K = cell(basix.CellType.interval)
    a = integral(coordinate(K)[0], K)
    points, weights = basix.make_quadrature(K.cell_type, 2)
    with pytest.raises(TypeError, match="single Integral"):
        apply_quadrature(a + a, points, weights)
    with pytest.raises(TypeError, match="single Integral"):
        apply_quadrature(form(a, ()), points, weights)
    with pytest.raises(ValueError, match="reference cell"):
        apply_quadrature(integral(num(1), EuclideanSpace(1)), points, weights)


def test_rule_shape_must_match_the_cell():
    expr = integral(num(1), cell(basix.CellType.triangle))
    with pytest.raises(ValueError, match="shape"):
        apply_quadrature(expr, [[0.5]], [1.0])


def test_weights_must_match_the_sample_index():
    K = cell(basix.CellType.interval)
    q = index("q")
    samples = evaluate_at_points(coordinate(K), [[0.5]], q)
    with pytest.raises(ValueError, match="point index"):
        weighted_sum(samples, [1.0, 2.0], q)
    with pytest.raises(ValueError, match="point index"):
        weighted_sum(samples, [1.0], index("another point index"))
    with pytest.raises(ValueError, match="real arrays"):
        weighted_sum(samples, [1j], q)
