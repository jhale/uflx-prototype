# Copyright (C) 2026 Jack S. Hale
#
# This file is part of uflx-core-experimental (https://fenics.github.io/uflx/language/)
#
# SPDX-License-Identifier:    MIT
"""Symbolic point evaluation on Basix reference cells."""

import numpy as np

from uflx.geometry import domain_map, euclidean_space, pull_back
from uflx.nodes import AbstractExpression, Index, Integral, ScalarType

from .nodes import BasixCell, table


def _real_array(data):
    if isinstance(data, AbstractExpression):
        if (
            data.domain is not None
            or data.argument_spaces
            or data.free_indices
            or data.scalar_type != ScalarType.real
        ):
            raise ValueError("operands must be domain-independent real arrays")
        return data
    array = np.asarray(data)
    if np.iscomplexobj(array):
        raise ValueError("operands must be real arrays")
    return table(array)


def _point_map(cell, points, q):
    if not isinstance(cell, BasixCell):
        raise ValueError("point evaluation needs a Basix reference cell")
    if not isinstance(q, Index):
        raise TypeError("expected an index for the point rows")
    if len(points.value_shape) != 2 or points.value_shape[1] != cell.dimension:
        raise ValueError("expected reference points of shape (n, tdim)")
    return domain_map(euclidean_space(0), points[q], cell)


def evaluate_at_points(expression, points, q, *, cell=None):
    """Evaluate a reference-cell expression at points[q], leaving q free.

    Points can be a concrete array or a real UFLx operand of shape (n, tdim).
    Supply the cell explicitly for a domain-independent expression. Other
    free indices and the expression's value shape are preserved.
    """
    cell = expression.domain if cell is None else cell
    mapping = _point_map(cell, _real_array(points), q)
    return Integral(pull_back(expression, mapping), mapping.parameter_domain)
