import basix

from uflx_spec_interpreter import RD, domain_map, index, index_sum, pull_back, tensor
from uflx_spec_interpreter._src.nodes import Float, Integral
from uflx_spec_interpreter._src.passes.pullbacks import expand_pull_backs
from uflx_spec_interpreter._src.substitution import map_fields
from uflx_toy.basix_nodes import BasixCell


def apply_quadrature(f, degree):
    # An integral over a reference cell becomes the weighted sum over the points of a rule.
    def rewrite(e):
        match map_fields(e, rewrite):
            case Integral(g, BasixCell() as K):
                return rule(g, K, degree)
            case Integral(g, RD(0)):
                # A point has the one weight 1.
                return g
            case e:
                return e

    return expand_pull_backs(rewrite(f))


def rule(g, K, degree):
    X, w = basix.make_quadrature(K.cell_type, degree)
    points = tensor(tuple(tensor(tuple(Float(float(x)) for x in p)) for p in X))
    weights = tensor(tuple(Float(float(x)) for x in w))
    q = index(("compiler.quadrature", g, K, degree))
    return index_sum(weights[q] * pull_back(g, domain_map(RD(0), points[q], K)), (q,))
