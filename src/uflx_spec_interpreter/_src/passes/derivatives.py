from ..nodes import DirectionalDerivative
from ..static import zero_like
from ..substitution import bottom_up, everywhere, memoized, occurs, substitute
from ..tuples import members
from .derivative_table import derivative_of_node


def deriv(f, x, a, h):
    # The rules give the derivative at x, so the point is substituted for x afterwards.
    xs, at, hs = members(x), members(a), members(h)
    out = memoized(lambda d, g: rules(d, g, xs, at, hs))(f)
    return out if a == x else substitute(out, dict(zip(members(x), members(a), strict=True)))


def rules(d, f, xs, a, h):
    # The leaf rows of apply_derivatives, then the derivatives of the node classes.
    constant = lambda c: not any(occurs(x, c) for x in xs)
    match f:
        case _ if constant(f):
            return zero_like(f, f.argument_spaces, *a, *h)
        case _ if f in xs:
            return h[xs.index(f)]
    return derivative_of_node(f, d, constant, [*xs, *a, *h])


def apply_derivatives(f):
    # Inner derivatives first, then the rule for the expression of each derivative.
    def evaluate(e):
        match e:
            case DirectionalDerivative(g, x, a, h):
                return deriv(g, x, a, h)
        return e

    return bottom_up(f, evaluate)


def in_n3(e):
    return everywhere(e, lambda n: not isinstance(n, DirectionalDerivative))
