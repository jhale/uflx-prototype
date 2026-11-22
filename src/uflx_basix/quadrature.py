"""Symbolic quadrature on Basix reference cells."""

import numpy as np

from uflx.algorithms import derived_index, derived_indices
from uflx.geometry import domain_map, pull_back
from uflx.nodes import EuclideanSpace, Integral, Zero
from uflx.tensors import component_tensor, index_sum, indexed

from .nodes import BasixCell, table


def apply_quadrature(integral, points, weights):
    """Sample one Basix reference-cell integral with an explicit quadrature rule.

    points has shape (n, cell dimension) and weights has shape (n,).
    The result is a UFLx weighted sum with the same value shape and free indices.
    """
    if not isinstance(integral, Integral):
        raise TypeError("expected a single Integral")
    K = integral.integration_domain
    if not isinstance(K, BasixCell):
        raise ValueError("expected an integral on a Basix reference cell")
    g = integral.expression
    points, weights = np.asarray(points), np.asarray(weights)
    if weights.ndim != 1 or points.shape != (weights.size, K.dimension):
        raise ValueError("expected weights of shape (n,) and reference points of shape (n, tdim)")
    if not weights.size:
        return Zero(
            integral.value_shape,
            integral.free_indices,
            integral.argument_spaces,
            integral.scalar_type,
        )
    points, weights = table(points), table(weights)
    q = derived_index("uflx_basix.quadrature", integral, points, weights)
    # IndexSum acts on a scalar, leaving the integrand's value axes and free indices intact.
    axes = derived_indices("uflx_basix.quadrature_component", (g,), len(g.value_shape))
    point = EuclideanSpace(0)
    sample_map = domain_map(point, points[q], K)
    sample = Integral(pull_back(indexed(g, axes), sample_map), point)
    return component_tensor(index_sum(weights[q] * sample, (q,)), axes)
