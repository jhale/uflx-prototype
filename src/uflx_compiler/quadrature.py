import basix
import numpy as np

from uflx_spec_interpreter import RD, domain_map, index_sum, pull_back, tensor
from uflx_spec_interpreter._src.names import _derived_index, _derived_indices
from uflx_spec_interpreter._src.nodes import Float, Integral
from uflx_spec_interpreter._src.passes.pullbacks import expand_pull_backs
from uflx_spec_interpreter._src.substitution import _bottom_up
from uflx_toy.basix_nodes import BasixCell


def apply_quadrature(f, degree):
    # An integral over a reference cell becomes the weighted sum over the points of a rule.
    def rewrite(e):
        match e:
            case Integral(g, BasixCell() as K):
                return rule(g, K, degree)
            case Integral(g, RD(0)):
                # A point has the one weight 1.
                return g
        return e

    return expand_pull_backs(_bottom_up(f, rewrite))


def rule(g, K, degree):
    if K.cell_type in (basix.CellType.quadrilateral, basix.CellType.hexahedron):
        return product_rule(g, K, degree)
    X, w = basix.make_quadrature(K.cell_type, degree)
    points, weights = constant(X), constant(w)
    # The index is named by the rule alone, so the geometry at the points is one node in every
    # integral, and the pull-back makes an inner sum over the same rule fresh.
    q = _derived_index("compiler.quadrature", K, degree)
    return index_sum(weights[q] * pull_back(g, domain_map(RD(0), points[q], K)), (q,))


def product_rule(g, K, degree):
    # Coordinate k of the points reads only its own index, which tabulation reads as separable.
    X, w = basix.make_quadrature(basix.CellType.interval, degree)
    X, w = constant(X[:, 0]), constant(w)
    q = _derived_indices("compiler.quadrature", (K, degree), K.dimension)
    points = tensor(tuple(X[i] for i in q))
    weight = w[q[0]]
    for i in q[1:]:
        weight = weight * w[i]
    return index_sum(weight * pull_back(g, domain_map(RD(0), points, K)), q)


def constant(values):
    # An array of numbers as a Tensor of Floats, or a Float.
    values = np.asarray(values)
    if values.ndim == 0:
        return Float(float(values))
    return tensor(tuple(map(constant, values)))
