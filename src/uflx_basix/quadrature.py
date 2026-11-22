"""Symbolic quadrature on Basix reference cells."""

from operator import index

import basix
import numpy as np

from uflx.algorithms import derived_index, derived_indices, map_values, memoized
from uflx.calculus import integral
from uflx.geometry import domain_map, pull_back
from uflx.nodes import EuclideanSpace, Float, Integral, Zero
from uflx.passes import expand_pull_backs
from uflx.tensors import component_tensor, index_sum, indexed, tensor

from .basix_nodes import BasixCell


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
        sampled = (
            explicit_rule(e.expression, K, *choice)
            if isinstance(choice, tuple)
            else rule(e.expression, K, choice)
        )
        # Keep the replacement domain-free while rebuilding its parents.
        result = integral(sampled, EuclideanSpace(0))
        return result if select is None else expand_pull_backs(result)

    result = memoized(visit)(f)
    # The fixed-degree pass also expands existing pull-backs outside the integrals.
    return expand_pull_backs(result) if select is None else result


def rule(g, K, degree):
    degree = index(degree)
    if K.dimension == 0:
        # A point has one quadrature point of weight one, with no coordinates.
        return pull_back(g, domain_map(EuclideanSpace(0), Zero((0,), (), (), "real"), K))
    if K.cell_type in (basix.CellType.quadrilateral, basix.CellType.hexahedron):
        return product_rule(g, K, degree)
    X, w = basix.make_quadrature(K.cell_type, degree)
    points, weights = constant(X), constant(w)
    # The index is named by the rule alone, so the geometry at the points is one node in every
    # integral, and the pull-back makes an inner sum over the same rule fresh.
    q = derived_index("compiler.quadrature", K, degree)
    return weighted_sum(g, K, points[q], weights[q], (q,))


def explicit_rule(g, K, points, weights):
    points, weights = np.asarray(points), np.asarray(weights)
    if weights.ndim != 1 or points.shape != (weights.size, K.dimension):
        raise ValueError("expected weights of shape (n,) and reference points of shape (n, tdim)")
    if not weights.size:
        return Zero(g.value_shape, g.free_indices, g.argument_spaces, g.scalar_type)
    points = constant(points) if K.dimension else Zero((weights.size, 0), (), (), "real")
    weights = constant(weights)
    q = derived_index("compiler.quadrature", K, points, weights)
    return weighted_sum(g, K, points[q], weights[q], (q,))


def product_rule(g, K, degree):
    # Coordinate k of the points reads only its own index, which tabulation reads as separable.
    X, w = basix.make_quadrature(basix.CellType.interval, degree)
    X, w = constant(X[:, 0]), constant(w)
    q = derived_indices("compiler.quadrature", (K, degree), K.dimension)
    points = tensor(tuple(X[i] for i in q))
    weight = w[q[0]]
    for i in q[1:]:
        weight = weight * w[i]
    return weighted_sum(g, K, points, weight, q)


def weighted_sum(g, K, points, weight, quadrature_indices):
    # IndexSum acts on a scalar, leaving the integrand's value axes and free indices intact.
    axes = derived_indices("compiler.quadrature_component", (g,), len(g.value_shape))
    value = pull_back(indexed(g, axes), domain_map(EuclideanSpace(0), points, K))
    return component_tensor(index_sum(weight * value, quadrature_indices), axes)


def constant(values):
    # An array of numbers as a Tensor of Floats, or a Float.
    values = np.asarray(values)
    if values.ndim == 0:
        return Float(float(values))
    return tensor(tuple(map(constant, values)))
