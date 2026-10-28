from ..node_derivatives import derivative_of_node
from ..nodes import DirectionalDerivative
from ..static import arity, zero_like
from ..substitution import expressions, map_fields, occurs, substitute


def deriv(f, x, a, h):
    # The rules give the derivative at x, so the point is substituted for x afterwards.
    out = rules(f, x, a, h)
    return out if a == x else substitute(out, {x: a})


def rules(f, x, a, h):
    # The leaf rows of apply_derivatives, then the derivatives of the node classes.
    match f:
        case _ if not occurs(x, f):
            return zero_like(f, arity(f), a, h)
        case _ if f == x:
            return h
    return derivative_of_node(f, lambda b: rules(b, x, a, h), lambda c: not occurs(x, c), [x, a, h])


def apply_derivatives(e):
    # Inner derivatives first, then the rule for the expression of each derivative.
    match map_fields(e, apply_derivatives):
        case DirectionalDerivative(f, x, a, h):
            return deriv(f, x, a, h)
        case e:
            return e


def in_n3(e):
    return not isinstance(e, DirectionalDerivative) and all(in_n3(c) for c in expressions(e))
