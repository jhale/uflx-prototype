"""Symbolic quadrature on Basix reference cells."""

from operator import index

import numpy as np

from uflx.algorithms import derived_index, derived_indices, map_values, memoized
from uflx.calculus import integral
from uflx.geometry import domain_map, pull_back
from uflx.nodes import EuclideanSpace, Integral, Zero
from uflx.passes import expand_pull_backs
from uflx.tensors import component_tensor, index_sum, indexed

from .nodes import BasixCell, table


def apply_quadrature(f, degree=None, *, select=None):
    """Apply a single degree or select a rule per Basix reference-cell integral.

    select receives each distinct original Integral in the expression once.
    Return a degree for the usual Basix rule, a (points, weights) pair in reference
    coordinates, or None to retain that integral. Points have shape (n, tdim)
    and weights have shape (n,). Nested integrals are visited first.
    """
    if (degree is None) == (select is None):
        raise TypeError("supply either a quadrature degree or a rule selector")

    def visit(walk, original):
        e = map_values(original, walk)
        if not isinstance(e, Integral) or not isinstance(e.integration_domain, BasixCell):
            return e
        choice = degree if select is None else select(original)
        if choice is None:
            return e
        K = e.integration_domain
        points, weights = choice if isinstance(choice, tuple) else K.quadrature(index(choice))
        sampled = explicit_rule(e.expression, K, points, weights)
        # Keep the replacement domain-free while rebuilding its parents.
        return expand_pull_backs(integral(sampled, EuclideanSpace(0)))

    return memoized(visit)(f)


def explicit_rule(g, K, points, weights):
    points, weights = np.asarray(points), np.asarray(weights)
    if weights.ndim != 1 or points.shape != (weights.size, K.dimension):
        raise ValueError("expected weights of shape (n,) and reference points of shape (n, tdim)")
    if not weights.size:
        return Zero(g.value_shape, g.free_indices, g.argument_spaces, g.scalar_type)
    points, weights = table(points), table(weights)
    q = derived_index("uflx_basix.quadrature", K, points, weights)
    # IndexSum acts on a scalar, leaving the integrand's value axes and free indices intact.
    axes = derived_indices("uflx_basix.quadrature_component", (g,), len(g.value_shape))
    value = pull_back(indexed(g, axes), domain_map(EuclideanSpace(0), points[q], K))
    return component_tensor(index_sum(weights[q] * value, (q,)), axes)
