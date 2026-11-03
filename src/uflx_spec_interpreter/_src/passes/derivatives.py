from ..linearity import dependence
from ..nodes import DirectionalDerivative, Function, ImaginaryUnit, Integer, Product
from ..static import complete_indices, zero_like
from ..substitution import bottom_up, everywhere, memoized, occurs, substitute
from ..tuples import each, members
from .actions import apply_actions
from .algebra import normalizer
from .derivative_table import derivative_of_node


def deriv(f, x, a, h, normalize=None):
    # The rules give the derivative at x, so the point is substituted for x afterwards.
    xs, at, hs = members(x), members(a), members(h)
    normalize = normalizer() if normalize is None else normalize
    out = memoized(lambda d, g: normalize(rules(d, g, xs, at, hs)))(f)
    out = substitute(out, dict(zip(xs, at, strict=True)))
    return complete_indices(DirectionalDerivative(f, x, a, h), out)


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
    normalize = normalizer()

    computed = {}

    def differential(g, x, a, h):
        key = (g, x, a, h)
        if key in computed:
            return computed[key]
        hs = members(h)
        if all(
            isinstance(v, Product)
            and isinstance(v.first, ImaginaryUnit)
            and isinstance(v.second, Function)
            for v in hs
        ):
            base = each(lambda v: v.second, h)
            # Only fresh directions can be treated as independent of the point/body.
            if not any(occurs(v, e) for v in members(base) for e in (g, *members(a))):
                result = differential(g, x, a, base)
                kind = dependence(result, base)
                if kind in ("zero", "complex-linear", "complex-antilinear"):
                    factor = ImaginaryUnit()
                    if kind == "complex-antilinear":
                        factor = Product(Integer(-1), factor)
                    out = normalize(Product(factor, result))
                    computed[key] = out
                    return out
        out = normalize(deriv(g, x, a, h, normalize))
        computed[key] = out
        return out

    def evaluate(e):
        match e:
            case DirectionalDerivative(g, x, a, h):
                return differential(g, x, a, h)
        return normalize(e)

    # Point substitution can expose actions whose application exposes derivatives.
    # Finish both operations before promising the derivative-free boundary.
    out = apply_actions(bottom_up(f, evaluate))
    return out if in_n3(out) else apply_derivatives(out)


def in_n3(e):
    return everywhere(e, lambda n: not isinstance(n, DirectionalDerivative))
