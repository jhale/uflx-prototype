from ..linearity import dependence
from ..nodes import Complex, DirectionalDerivative, Function, Integer, Product
from ..static import complete_indices, promote, zero_like
from ..substitution import bottom_up, everywhere, memoized, occurs, substitute_values
from ..tuples import each, members
from .actions import apply_actions
from .algebra import normalizer
from .derivative_table import derivative_of_node


def deriv(f, x, a, h, normalize=None):
    # The rules give the derivative at x, so the point is substituted for x afterwards.
    xs, at, hs = members(x), members(a), members(h)
    at = tuple(promote(v, p) for v, p in zip(xs, at, strict=True))
    hs = tuple(promote(v, h) for v, h in zip(xs, hs, strict=True))
    normalize = normalizer() if normalize is None else normalize
    variables = tuple(Function(v.space, ("uflx.derivative", v, f, x, a, h)) for v in xs)
    body = substitute_values(f, dict(zip(xs, variables, strict=True)))
    out = memoized(lambda d, g: normalize(rules(d, g, variables, at, hs)))(body)
    out = substitute_values(out, dict(zip(variables, at, strict=True)))
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
            and v.first == Complex(Integer(0), Integer(1))
            and isinstance(v.second, Function)
            for v in hs
        ):
            base = each(lambda v: v.second, h)
            # Only fresh directions can be treated as independent of the point/body.
            if not any(occurs(v, e) for v in members(base) for e in (g, *members(a))):
                result = differential(g, x, a, base)
                kind = dependence(result, base)
                if kind in ("zero", "complex-linear", "complex-antilinear"):
                    factor = Complex(Integer(0), Integer(1))
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
