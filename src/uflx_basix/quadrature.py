# Copyright (C) 2026 Jack S. Hale
#
# This file is part of uflx-core-experimental (https://fenics.github.io/uflx/language/)
#
# SPDX-License-Identifier:    MIT
"""Symbolic quadrature on Basix reference cells."""

import numpy as np

from uflx.algorithms import derived_index, derived_indices
from uflx.geometry import domain_map, euclidean_space, pull_back
from uflx.nodes import AbstractExpression, Integral, ScalarType, Zero
from uflx.tensors import component_tensor, index_sum, indexed

from .nodes import BasixCell, table


def apply_quadrature(integral, points, weights):
    """Sample one Basix reference-cell integral with an explicit quadrature rule.

    Pass concrete arrays or domain-independent real expressions with shapes
    (n, cell dimension) and (n,). The shared extent n may be symbolic.
    The result is a UFLx weighted sum with the same value shape and free indices.
    """
    if not isinstance(integral, Integral):
        raise TypeError("expected a single Integral")
    K = integral.integration_domain
    if not isinstance(K, BasixCell):
        raise ValueError("expected an integral on a Basix reference cell")
    g = integral.expression

    def operand(data):
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

    points, weights = operand(points), operand(weights)
    if len(weights.value_shape) != 1 or points.value_shape != (*weights.value_shape, K.dimension):
        raise ValueError("expected weights of shape (n,) and reference points of shape (n, tdim)")
    if weights.value_shape == (0,):
        return Zero(
            integral.value_shape,
            integral.free_indices,
            integral.argument_spaces,
            integral.scalar_type,
        )
    q = derived_index("uflx_basix.quadrature", integral, points, weights)
    # IndexSum acts on a scalar, leaving the integrand's value axes and free indices intact.
    axes = derived_indices("uflx_basix.quadrature_component", (g,), len(g.value_shape))
    point = euclidean_space(0)
    sample_map = domain_map(point, points[q], K)
    sample = Integral(pull_back(indexed(g, axes), sample_map), point)
    return component_tensor(index_sum(weights[q] * sample, (q,)), axes)
