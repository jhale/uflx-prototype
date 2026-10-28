import basix

from uflx_spec_interpreter import RD, domain_map, index, index_sum, pull_back, tensor
from uflx_spec_interpreter._src.nodes import Float, Integral
from uflx_spec_interpreter._src.passes.pullbacks import expand_pull_backs
from uflx_spec_interpreter._src.substitution import bottom_up
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

    return expand_pull_backs(bottom_up(f, rewrite))


def rule(g, K, degree):
    if K.cell_type in PRODUCTS:
        return product_rule(g, K, degree)
    X, w = basix.make_quadrature(K.cell_type, degree)
    points = tensor(tuple(tensor(tuple(Float(float(x)) for x in p)) for p in X))
    weights = constants(w)
    q = index(("compiler.quadrature", g, K, degree))
    return index_sum(weights[q] * pull_back(g, domain_map(RD(0), points[q], K)), (q,))


def product_rule(g, K, degree):
    # Coordinate k of the points reads only its own index, which tabulation reads as separable.
    X, w = basix.make_quadrature(basix.CellType.interval, degree)
    X, w = constants(X[:, 0]), constants(w)
    q = tuple(index(("compiler.quadrature", g, K, degree, k)) for k in range(K.dimension))
    points = tensor(tuple(X[i] for i in q))
    weight = w[q[0]]
    for i in q[1:]:
        weight = weight * w[i]
    return index_sum(weight * pull_back(g, domain_map(RD(0), points, K)), q)


PRODUCTS = (basix.CellType.quadrilateral, basix.CellType.hexahedron)


def constants(values):
    return tensor(tuple(Float(float(x)) for x in values))
