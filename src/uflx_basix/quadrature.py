# Copyright (C) 2026 Jack S. Hale
#
# This file is part of uflx-core-experimental (https://fenics.github.io/uflx/language/)
#
# SPDX-License-Identifier:    MIT
"""Symbolic point evaluation and quadrature on Basix reference cells."""

import numpy as np

from uflx.algorithms import derived_index, derived_indices
from uflx.geometry import domain_map, euclidean_space, pull_back
from uflx.nodes import AbstractExpression, Index, Integral, ScalarType, Zero
from uflx.tensors import component_tensor, index_sum, indexed

from .nodes import BasixCell, table


def _real_array(data):
    if isinstance(data, AbstractExpression):
        if (
            data.domain is not None
            or data.argument_spaces
            or data.free_indices
            or data.scalar_type != ScalarType.real
        ):
            raise ValueError("quadrature operands must be domain-independent real arrays")
        return data
    array = np.asarray(data)
    if np.iscomplexobj(array):
        raise ValueError("quadrature operands must be real arrays")
    return table(array)


def evaluate_at_points(expression, points, q, *, cell=None):
    """Evaluate a reference-cell expression at points[q], leaving q free.

    Points can be a concrete array or a real UFLx operand of shape (n, tdim).
    Supply the cell explicitly for a domain-independent expression. Other
    free indices and the expression's value shape are preserved.
    """
    cell = expression.domain if cell is None else cell
    if not isinstance(cell, BasixCell):
        raise ValueError("point evaluation needs a Basix reference cell")
    if not isinstance(q, Index):
        raise TypeError("expected an index for the point rows")
    points = _real_array(points)
    if len(points.value_shape) != 2 or points.value_shape[1] != cell.dimension:
        raise ValueError("expected reference points of shape (n, tdim)")
    point = euclidean_space(0)
    return Integral(pull_back(expression, domain_map(point, points[q], cell)), point)


def weighted_sum(samples, weights, q):
    """Sum weights[q] times samples over q, preserving all other axes."""
    weights = _real_array(weights)
    if weights.value_shape != (dict(samples.free_indices).get(q),):
        raise ValueError("expected weights of shape (n,) matching the samples' point index")
    axes = derived_indices("uflx_basix.quadrature_component", (samples,), len(samples.value_shape))
    total = component_tensor(index_sum(weights[q] * indexed(samples, axes), (q,)), axes)
    if weights.value_shape == (0,):
        return Zero(total.value_shape, total.free_indices, total.argument_spaces, total.scalar_type)
    return total


def apply_quadrature(integral, points, weights):
    """Evaluate one reference-cell integral by sampling its integrand and summing with weights."""
    if not isinstance(integral, Integral):
        raise TypeError("expected a single Integral")
    q = derived_index("uflx_basix.quadrature", integral)
    samples = evaluate_at_points(integral.expression, points, q, cell=integral.integration_domain)
    return weighted_sum(samples, weights, q)
