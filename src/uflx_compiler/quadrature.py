import basix
import numpy as np

from uflx import (
    EuclideanSpace,
    component_tensor,
    domain_map,
    index_sum,
    indexed,
    integral,
    pull_back,
    tensor,
)
from uflx._src.algorithms.names import derived_index, derived_indices
from uflx._src.algorithms.traversal import bottom_up
from uflx._src.nodes import Float, Integral, Zero
from uflx._src.passes.pullbacks import expand_pull_backs
from uflx_basix.basix_nodes import BasixCell


def apply_quadrature(f, degree):
    # An integral over a reference cell becomes the weighted sum over the points of a rule.
    def rewrite(e):
        match e:
            case Integral(g, BasixCell() as K):
                # Keep the result domain-free while rebuilding enclosing fields.
                # Point integrals are removed after tabulation, when their operands
                # are domain-free arrays rather than fields on EuclideanSpace(0).
                return integral(rule(g, K, degree), EuclideanSpace(0))
        return e

    return expand_pull_backs(bottom_up(f, rewrite))


def rule(g, K, degree):
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
